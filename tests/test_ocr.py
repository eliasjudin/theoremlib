"""
Test suite for OCR module functionality
"""

import pytest
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch

from ..ocr.ocr_module import OCRProcessor
from .test_core import sample_metadata

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create minimal PDF with text layer
        tmp.write(b"%PDF-1.7\n%Test PDF with text layer")
        tmp_path = tmp.name
    yield Path(tmp_path)
    os.unlink(tmp_path)

@pytest.mark.asyncio
async def test_ocr_text_layer_detection(sample_pdf):
    """Test detection of existing text layer"""
    processor = OCRProcessor(api_key="test_key")
    has_text = await processor.check_text_layer(sample_pdf)
    assert has_text is True

@pytest.mark.asyncio
async def test_ocr_processing():
    """Test OCR processing with mock API response"""
    mock_response = {
        "text": "Theorem 1. Example theorem statement\nProof: Example proof",
        "confidence": 0.95
    }
    
    with patch("ocr.ocr_module.OCRProcessor._call_ocr_api", 
              return_value=mock_response):
        processor = OCRProcessor(api_key="test_key")
        result = await processor.process_page(
            page_image=b"test_image",
            page_number=1
        )
        assert result["text"] is not None
        assert result["confidence"] > 0.9

@pytest.mark.asyncio
async def test_ocr_retries():
    """Test OCR retry logic for low confidence results"""
    responses = [
        {"text": "Low quality text", "confidence": 0.4},
        {"text": "Better quality text", "confidence": 0.9}
    ]
    
    mock_call = Mock(side_effect=responses)
    with patch("ocr.ocr_module.OCRProcessor._call_ocr_api", mock_call):
        processor = OCRProcessor(api_key="test_key")
        result = await processor.process_page(
            page_image=b"test_image",
            page_number=1,
            min_confidence=0.8
        )
        assert mock_call.call_count == 2
        assert result["confidence"] > 0.8

@pytest.mark.asyncio
async def test_ocr_parameter_tuning():
    """Test OCR parameter tuning for optimal results"""
    processor = OCRProcessor(api_key="test_key")
    params = processor.get_optimal_parameters(
        sample_quality=0.5,
        page_type="math"
    )
    assert "math_mode" in params
    assert params.get("enhance_formulas", False) is True

@pytest.mark.asyncio
async def test_ocr_error_handling():
    """Test error handling during OCR processing"""
    with patch("ocr.ocr_module.OCRProcessor._call_ocr_api", 
              side_effect=Exception("API Error")):
        processor = OCRProcessor(api_key="test_key")
        with pytest.raises(Exception) as exc_info:
            await processor.process_page(
                page_image=b"test_image",
                page_number=1
            )
        assert "API Error" in str(exc_info.value)