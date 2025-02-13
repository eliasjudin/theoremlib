"""
PDF Ingestion Module

Handles the ingestion of mathematical PDFs, validation of LaTeX text layers,
and preparation for content extraction.
"""

import fitz  # PyMuPDF
import os
import logging
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator, root_validator
import mimetypes
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    """Metadata for ingested documents"""
    source: str
    title: str
    author: str
    page_count: int
    theorem_number: Optional[str] = ""
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    file_hash: Optional[str] = None
    
    @validator('upload_time')
    def ensure_utc(cls, v):
        """Ensure upload_time is in UTC"""
        return v.astimezone(datetime.timezone.utc)
    
    @root_validator
    def validate_metadata(cls, values):
        """Validate metadata fields"""
        if values.get('page_count', 0) <= 0:
            raise ValueError("Page count must be positive")
        if not values.get('title', '').strip():
            raise ValueError("Title cannot be empty")
        return values

class PDFMetadata(BaseModel):
    """Extended metadata model for ingested PDFs including technical details"""
    title: str
    authors: list[str]
    year: Optional[int]
    source: str
    license: str
    has_text_layer: bool
    needs_ocr: bool
    page_count: int
    document_metadata: Optional[DocumentMetadata] = None

class PDFIngestor:
    def __init__(self, storage_path: Path):
        """
        Initialize the PDF ingestor with storage configuration.
        
        Args:
            storage_path: Base path for PDF storage
        """
        self.storage_path = storage_path
        self.ingested_files_path = storage_path / "ingested_files"
        self.ingested_files_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Track processed files with thread-safe lock
        self.processed_hashes = set()
        self._hash_lock = asyncio.Lock()
        
        # Load existing hashes from processed files
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Load hashes of already processed files"""
        for file_path in self.ingested_files_path.glob("*"):
            if file_path.is_file():
                file_hash = self._compute_file_hash(file_path)
                self.processed_hashes.add(file_hash)

    def _setup_logging(self):
        """Configure logging with rotation and formatting"""
        log_path = self.storage_path / "logs"
        log_path.mkdir(exist_ok=True)
        
        fh = logging.FileHandler(log_path / "ingest.log")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of file contents to detect duplicates
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            str: Hex digest of file hash
        """
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _validate_pdf(self, file_path: Path) -> None:
        """
        Validate PDF file format and accessibility
        
        Args:
            file_path: Path to the PDF file to validate
            
        Raises:
            ValueError: If file is not a valid PDF
            FileNotFoundError: If file cannot be accessed
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        mime_type = mimetypes.guess_type(file_path)[0]
        if mime_type != 'application/pdf':
            raise ValueError(f"Invalid file type: {mime_type}, expected PDF")
            
        try:
            doc = fitz.open(file_path)
            doc.close()
        except Exception as e:
            raise ValueError(f"Invalid or corrupted PDF file: {e}")

    def validate_text_layer(self, pdf_path: Path) -> Tuple[bool, float]:
        """
        Verifies if PDF has a complete LaTeX text layer.
        Returns (has_text_layer, confidence_score)
        """
        doc = fitz.open(pdf_path)
        text_coverage = []
        
        for page in doc:
            text = page.get_text("text")
            blocks = page.get_text("blocks")
            # Calculate text coverage score for the page
            text_coverage.append(len(blocks) > 0 and len(text.strip()) > 0)
        
        doc.close()
        coverage_score = sum(text_coverage) / len(text_coverage)
        return coverage_score > 0.9, coverage_score

    def _store_pdf(self, source_path: Path) -> Path:
        """
        Store a PDF file with additional validation
        
        Args:
            source_path: Path to the source PDF file
            
        Returns:
            Path to the stored file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_hash = self._compute_file_hash(source_path)[:8]
        stored_name = f"{source_path.stem}_{timestamp}_{file_hash}{source_path.suffix}"
        target_path = self.ingested_files_path / stored_name
        
        # Ensure we don't overwrite existing files
        while target_path.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            stored_name = f"{source_path.stem}_{timestamp}_{file_hash}{source_path.suffix}"
            target_path = self.ingested_files_path / stored_name
        
        shutil.copy2(source_path, target_path)
        return target_path

    def _log_metadata(self, metadata: Union[DocumentMetadata, PDFMetadata]):
        """
        Log document metadata to both console and file.
        
        Args:
            metadata: Document or PDF metadata to log
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata.dict(exclude_none=True)
        }
        logger.info(f"Document ingested: {log_entry}")

    @asynccontextmanager
    async def _check_duplicate(self, file_hash: str):
        """Thread-safe duplicate checking"""
        async with self._hash_lock:
            if file_hash in self.processed_hashes:
                raise ValueError("This PDF has already been processed")
            self.processed_hashes.add(file_hash)
            try:
                yield
            except Exception:
                self.processed_hashes.remove(file_hash)
                raise

    async def ingest_pdf(
        self,
        pdf_path: Union[Path, str],
        metadata: Dict,
        doc_metadata: Optional[DocumentMetadata] = None
    ) -> PDFMetadata:
        """
        Ingest a PDF file with enhanced validation and deduplication
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Basic metadata dictionary
            doc_metadata: Optional DocumentMetadata instance
            
        Returns:
            PDFMetadata instance with complete metadata
        """
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)

        # Validate PDF file
        self._validate_pdf(pdf_path)
        
        # Compute file hash and check for duplicates
        file_hash = self._compute_file_hash(pdf_path)
        
        async with self._check_duplicate(file_hash):
            # Add file hash to metadata
            if doc_metadata:
                doc_metadata.file_hash = file_hash

            # Process the PDF
            try:
                has_text, confidence = self.validate_text_layer(pdf_path)
                stored_path = self._store_pdf(pdf_path)
                
                doc = fitz.open(pdf_path)
                pdf_metadata = PDFMetadata(
                    title=metadata.get("title", pdf_path.stem),
                    authors=metadata.get("authors", []),
                    year=metadata.get("year"),
                    source=metadata.get("source", "unknown"),
                    license=metadata.get("license", "CC0"),
                    has_text_layer=has_text,
                    needs_ocr=not has_text,
                    page_count=len(doc),
                    document_metadata=doc_metadata
                )
                
                # Log metadata
                self._log_metadata(pdf_metadata)
                doc.close()
                
                return pdf_metadata
                
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
                # Clean up any partially processed files
                if 'stored_path' in locals() and stored_path.exists():
                    stored_path.unlink()
                raise