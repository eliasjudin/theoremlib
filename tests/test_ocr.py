"""
Tests for OCR and LaTeX layer verification functionality
"""

import pytest
from pathlib import Path
import asyncio
import fitz
from unittest.mock import Mock, patch
from google.cloud import vision

from ..ocr.ocr_module import (
    OCRProcessor,
    OCRResult,
    OCRParameters,
    ExtractionMethod
)

@pytest.fixture
def sample_latex_pdf(tmp_path):
    """Create a sample PDF with LaTeX content"""
    pdf_path = tmp_path / "latex_sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    
    # Add some LaTeX-like content
    latex_text = r"""
    \begin{theorem}
    For any real number x, $x^2 \geq 0$.
    \end{theorem}
    
    \begin{proof}
    Let x be any real number. Then...
    \end{proof}
    """
    page.insert_text((50, 50), latex_text)
    doc.save(pdf_path)
    doc.close()
    return pdf_path

@pytest.fixture
def sample_scanned_pdf(tmp_path):
    """Create a sample PDF simulating a scanned document"""
    pdf_path = tmp_path / "scanned_sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    
    # Add image-based content (simulating a scan)
    page.insert_image((0, 0), pixmap=fitz.Pixmap(fitz.csRGB, (100, 100), b"\xff" * 100 * 100 * 3))
    doc.save(pdf_path)
    doc.close()
    return pdf_path

@pytest.fixture
def mock_vision_client():
    """Mock Google Cloud Vision client"""
    with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client:
        # Create mock response
        mock_response = Mock()
        mock_page = Mock()
        mock_block = Mock()
        mock_block.confidence = 0.95
        mock_block.text = "Sample text"
        mock_block.bounding_box.vertices = [
            Mock(x=0, y=0),
            Mock(x=100, y=0),
            Mock(x=100, y=100),
            Mock(x=0, y=100)
        ]
        mock_page.blocks = [mock_block]
        mock_response.pages = [mock_page]
        mock_response.full_text_annotation.text = "Sample text"
        
        mock_client.return_value.document_text_detection.return_value = mock_response
        yield mock_client

@pytest.mark.asyncio
async def test_latex_layer_detection(sample_latex_pdf):
    """Test detection of LaTeX text layer"""
    processor = OCRProcessor("dummy_key")
    has_latex, confidence = processor.has_complete_latex_layer(sample_latex_pdf)
    
    assert has_latex
    assert confidence > 0.9
    assert confidence <= 1.0

@pytest.mark.asyncio
async def test_latex_extraction(sample_latex_pdf):
    """Test extraction from LaTeX layer"""
    processor = OCRProcessor("dummy_key")
    results = await processor.extract_from_latex_layer(sample_latex_pdf)
    
    assert len(results) > 0
    assert all(r.extraction_method == ExtractionMethod.LATEX for r in results)
    assert all(r.confidence == 1.0 for r in results)
    assert "theorem" in results[0].text_content.lower()
    assert "proof" in results[0].text_content.lower()

@pytest.mark.asyncio
async def test_ocr_fallback(sample_scanned_pdf, mock_vision_client):
    """Test OCR fallback for scanned documents"""
    processor = OCRProcessor("dummy_key")
    results = await processor.process_pdf(sample_scanned_pdf)
    
    assert len(results) > 0
    assert any(r.extraction_method == ExtractionMethod.OCR for r in results)
    assert all(r.confidence > 0 for r in results)

@pytest.mark.asyncio
async def test_ocr_retry_logic(sample_scanned_pdf, mock_vision_client):
    """Test OCR retry logic with parameter adjustment"""
    # Make first attempt fail
    mock_vision_client.return_value.document_text_detection.side_effect = [
        Exception("First attempt failed"),
        Mock(
            pages=[Mock(
                blocks=[Mock(confidence=0.95, text="Retry successful")],
                full_text_annotation=Mock(text="Retry successful")
            )]
        )
    ]
    
    processor = OCRProcessor("dummy_key")
    result = await processor.process_page(sample_scanned_pdf, 0)
    
    assert not result.requires_review
    assert "Retry successful" in result.text_content
    assert result.confidence > 0.9

@pytest.mark.asyncio
async def test_manual_review_flagging(sample_scanned_pdf, mock_vision_client):
    """Test flagging for manual review after failed attempts"""
    # Make all attempts fail with low confidence
    mock_vision_client.return_value.document_text_detection.return_value = Mock(
        pages=[Mock(
            blocks=[Mock(confidence=0.5, text="Low confidence text")],
            full_text_annotation=Mock(text="Low confidence text")
        )]
    )
    
    processor = OCRProcessor("dummy_key", confidence_threshold=0.9)
    result = await processor.process_page(sample_scanned_pdf, 0)
    
    assert result.requires_review
    assert result.confidence < 0.9
    assert result.extraction_method == ExtractionMethod.MANUAL

@pytest.mark.asyncio
async def test_ocr_parameters_adjustment():
    """Test OCR parameters adjustment logic"""
    params = OCRParameters()
    initial_dpi = params.dpi
    
    params.adjust_for_retry()
    
    assert params.dpi > initial_dpi
    assert params.enhance_math
    assert params.page_segmentation_mode == 3  # Automatic mode