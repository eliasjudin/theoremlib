"""
OCR Module

Handles OCR processing and LaTeX layer verification for mathematical PDFs,
with support for Google Cloud Vision (Gemini 2.0) extraction and automatic retry logic.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
from google.cloud import vision
from pdf2image import convert_from_path
import asyncio
from pydantic import BaseModel
import fitz  # PyMuPDF
import logging
import tempfile
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
from typing import Iterator, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExtractionMethod(str, Enum):
    LATEX = "latex"
    OCR = "ocr"
    MANUAL = "manual_review"

@dataclass
class OCRParameters:
    """Parameters for OCR processing"""
    dpi: int = 300
    language_hints: List[str] = None
    enhance_math: bool = True
    page_segmentation_mode: int = 6  # Assume uniform text with math formulas
    
    def adjust_for_retry(self):
        """Adjust parameters for retry attempt"""
        self.dpi = 600  # Increase resolution
        self.page_segmentation_mode = 3  # Switch to fully automatic mode
        self.enhance_math = True

class OCRResult(BaseModel):
    """Model for OCR processing results"""
    page_number: int
    text_content: str
    confidence: float
    spatial_layout: Dict
    requires_review: bool
    extraction_method: ExtractionMethod

class LatexExtractionError(Exception):
    """Custom exception for LaTeX extraction failures"""
    pass

class OCRProcessor:
    def __init__(
        self,
        api_key: str,
        max_retries: int = 2,
        confidence_threshold: float = 0.9,
        latex_block_size: int = 1024  # Size of text blocks for LaTeX parsing
    ):
        self.client = vision.ImageAnnotatorClient()
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold
        self.parameters = OCRParameters()
        self.latex_block_size = latex_block_size

    @contextmanager
    def _safe_document_handling(self, pdf_path: Path) -> Iterator[fitz.Document]:
        """Safely handle PDF document operations with proper cleanup"""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            yield doc
        except Exception as e:
            logger.error(f"Error handling PDF document: {e}")
            raise LatexExtractionError(f"Failed to process PDF: {str(e)}")
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as e:
                    logger.warning(f"Error closing PDF document: {e}")

    def has_complete_latex_layer(self, pdf_path: Path) -> Tuple[bool, float]:
        """
        Check if PDF has a complete LaTeX text layer
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (has_latex, confidence_score)
        """
        doc = fitz.open(pdf_path)
        page_scores = []
        
        try:
            for page in doc:
                text = page.get_text("text")
                blocks = page.get_text("dict")["blocks"]
                
                # Check for LaTeX markers and math environments
                has_latex_markers = any(
                    "\\begin{" in text or "\\end{" in text or "$" in text
                )
                
                # Calculate text coverage and structure score
                text_length = len(text.strip())
                blocks_count = len(blocks)
                
                # Compute page score based on multiple factors
                page_score = 0.0
                if text_length > 100:  # Minimum text length threshold
                    page_score += 0.4
                if blocks_count > 5:    # Minimum structure threshold
                    page_score += 0.3
                if has_latex_markers:   # LaTeX presence bonus
                    page_score += 0.3
                    
                page_scores.append(page_score)
                
            # Calculate overall confidence
            confidence = sum(page_scores) / len(page_scores) if page_scores else 0.0
            has_latex = confidence >= self.confidence_threshold
            
            return has_latex, confidence
            
        finally:
            doc.close()

    async def extract_from_latex_layer(self, pdf_path: Path) -> List[OCRResult]:
        """Extract text directly from PDF's LaTeX layer with enhanced error handling"""
        results = []
        
        with self._safe_document_handling(pdf_path) as doc:
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    result = await self._process_latex_page(page, page_num)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    # Fall back to OCR for failed pages
                    logger.info(f"Falling back to OCR for page {page_num}")
                    ocr_result = await self._ocr_fallback_for_page(pdf_path, page_num)
                    results.append(ocr_result)
                    
        return results

    async def _process_latex_page(self, page: fitz.Page, page_num: int) -> OCRResult:
        """Process a single page's LaTeX content with validation"""
        text = page.get_text("text")
        blocks = page.get_text("dict")["blocks"]
        
        if not text.strip() or not blocks:
            raise LatexExtractionError("Empty or invalid LaTeX content")
            
        # Validate LaTeX structure
        if not self._validate_latex_structure(text):
            raise LatexExtractionError("Invalid LaTeX structure")
            
        spatial_layout = {
            "blocks": [
                {
                    "bbox": block["bbox"],
                    "type": block["type"],
                    "text": block.get("text", ""),
                    "lines": len(block.get("lines", []))
                }
                for block in blocks
            ]
        }
        
        return OCRResult(
            page_number=page_num,
            text_content=text,
            confidence=self._calculate_latex_confidence(text, blocks),
            spatial_layout=spatial_layout,
            requires_review=False,
            extraction_method=ExtractionMethod.LATEX
        )

    def _validate_latex_structure(self, text: str) -> bool:
        """Validate LaTeX structure for common issues"""
        # Check for balanced environments
        env_starts = text.count("\\begin{")
        env_ends = text.count("\\end{")
        if env_starts != env_ends:
            return False
            
        # Check for balanced math delimiters
        dollar_count = text.count("$")
        if dollar_count % 2 != 0:
            return False
            
        # Check for basic theorem-proof structure
        has_theorem = any(
            marker in text
            for marker in ["theorem", "lemma", "proposition", "corollary"]
        )
        has_proof = "proof" in text.lower()
        
        return has_theorem or has_proof

    def _calculate_latex_confidence(self, text: str, blocks: List[dict]) -> float:
        """Calculate confidence score for LaTeX extraction"""
        # Start with base confidence
        confidence = 0.8
        
        # Adjust based on LaTeX markers
        if "\\begin{" in text and "\\end{" in text:
            confidence += 0.1
            
        # Check for mathematical content
        if "$" in text or "\\[" in text:
            confidence += 0.05
            
        # Check block structure
        if len(blocks) >= 3:  # Typical theorem-proof structure
            confidence += 0.05
            
        return min(confidence, 1.0)

    async def _ocr_fallback_for_page(self, pdf_path: Path, page_num: int) -> OCRResult:
        """Handle OCR fallback for failed LaTeX extraction"""
        logger.info(f"Using OCR fallback for page {page_num}")
        
        # Convert single page to image
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=self.parameters.dpi
            )
            
            if not images:
                return OCRResult(
                    page_number=page_num,
                    text_content="",
                    confidence=0.0,
                    spatial_layout={},
                    requires_review=True,
                    extraction_method=ExtractionMethod.MANUAL
                )
                
            temp_path = temp_dir / f"page_{page_num}.png"
            images[0].save(temp_path, format="PNG", optimize=True)
            
            return await self.process_page(temp_path, page_num)

    async def process_page(
        self,
        image_path: Path,
        page_num: int,
        attempt: int = 0
    ) -> OCRResult:
        """Process a single page with OCR and retry capability"""
        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            
            # Configure OCR parameters based on attempt number
            if attempt > 0:
                self.parameters.adjust_for_retry()
                
            # Create image context with current parameters
            image_context = vision.ImageContext(
                language_hints=self.parameters.language_hints or ["en"],
                text_detection_params=vision.TextDetectionParams(
                    enable_text_detection_confidence_score=True
                )
            )

            response = self.client.document_text_detection(
                image=image,
                image_context=image_context
            )

            text = response.full_text_annotation.text
            
            # Calculate confidence score
            blocks = response.pages[0].blocks if response.pages else []
            confidence = (
                sum(block.confidence for block in blocks) / len(blocks)
                if blocks else 0.0
            )

            # Extract spatial layout
            layout = self._extract_spatial_layout(blocks)
            
            requires_review = confidence < self.confidence_threshold
            
            return OCRResult(
                page_number=page_num,
                text_content=text,
                confidence=confidence,
                spatial_layout=layout,
                requires_review=requires_review,
                extraction_method=ExtractionMethod.OCR
            )

        except Exception as e:
            logger.error(f"OCR processing failed on attempt {attempt}: {str(e)}")
            if attempt < self.max_retries:
                logger.info(f"Retrying with adjusted parameters...")
                await asyncio.sleep(1)
                return await self.process_page(image_path, page_num, attempt + 1)
            else:
                return OCRResult(
                    page_number=page_num,
                    text_content="",
                    confidence=0.0,
                    spatial_layout={},
                    requires_review=True,
                    extraction_method=ExtractionMethod.MANUAL
                )

    async def process_pdf(self, pdf_path: Path) -> List[OCRResult]:
        """Process a PDF document with automatic method selection"""
        has_latex, latex_confidence = self.has_complete_latex_layer(pdf_path)
        
        if has_latex:
            logger.info(f"Using LaTeX layer extraction (confidence: {latex_confidence:.2f})")
            return await self.extract_from_latex_layer(pdf_path)
        
        logger.info("LaTeX layer unavailable or incomplete, falling back to OCR")
        return await self._process_with_ocr(pdf_path)

    async def _process_with_ocr(self, pdf_path: Path) -> List[OCRResult]:
        """Process PDF using OCR with retry logic"""
        # Convert PDF to images
        images = convert_from_path(
            pdf_path,
            dpi=self.parameters.dpi,
            use_pdftocairo=True  # Better quality for mathematical documents
        )
        
        results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Process each page
            for i, image in enumerate(images):
                temp_path = temp_dir / f"page_{i}.png"
                image.save(temp_path, format="PNG", optimize=True)
                
                result = await self.process_page(temp_path, i)
                results.append(result)
                
        # Check if manual review is needed
        if any(r.requires_review for r in results):
            logger.warning(f"Manual review required for {pdf_path}")
            
        return results

    @staticmethod
    def _extract_spatial_layout(blocks) -> Dict:
        """Extract spatial layout information from OCR blocks"""
        return {
            "blocks": [
                {
                    "bbox": {
                        "x1": block.bounding_box.vertices[0].x,
                        "y1": block.bounding_box.vertices[0].y,
                        "x2": block.bounding_box.vertices[2].x,
                        "y2": block.bounding_box.vertices[2].y,
                    },
                    "text": block.text,
                    "confidence": block.confidence,
                    "symbols": len(block.text)
                }
                for block in blocks
            ]
        }