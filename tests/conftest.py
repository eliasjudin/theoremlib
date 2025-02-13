"""
Test configuration and fixtures for the theorem database system
"""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import asyncio
from pathlib import Path
import json
import tempfile
import fitz
from unittest.mock import Mock, patch
from google.cloud import vision

from ..database.models import Base
from ..database.database_loader import DatabaseLoader
from ..ingest.ingest import PDFIngestor
from ..extraction.content_extraction import ContentExtractor
from ..annotation.metadata_annotation import MetadataAnnotator
from ..ocr.ocr_module import OCRProcessor

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_storage_path():
    """Provide a temporary storage path for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir)
        (storage_path / "ingested_files").mkdir(exist_ok=True)
        yield storage_path

@pytest.fixture(scope="session")
def test_pdf_path(test_storage_path):
    """Create and provide a test PDF file"""
    pdf_path = test_storage_path / "test_theorem.pdf"
    # Create a minimal valid PDF for testing
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\ntrailer\n<</Root 1 0 R>>\n%%EOF")
    return pdf_path

@pytest.fixture(scope="session")
async def test_db():
    """Create a test database and provide test session"""
    test_engine = create_async_engine(
        "postgresql+asyncpg://theoremlib:theoremlib@localhost:5432/theoremlib_test"
    )
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    yield async_session
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
def test_ontology():
    """Provide test mathematical ontology"""
    return {
        "test_concept": {
            "id": "test.concept",
            "name": "Test Concept",
            "category": "test",
            "parent_concepts": [],
            "related_concepts": []
        }
    }

@pytest.fixture
def pdf_ingestor(test_storage_path):
    """Provide configured PDF ingestor"""
    return PDFIngestor(test_storage_path)

@pytest.fixture
def content_extractor():
    """Provide configured content extractor"""
    return ContentExtractor()

@pytest.fixture
def metadata_annotator(test_ontology, test_storage_path):
    """Provide configured metadata annotator"""
    ontology_path = test_storage_path / "test_ontology.json"
    with open(ontology_path, "w") as f:
        json.dump(test_ontology, f)
    return MetadataAnnotator(ontology_path)

@pytest.fixture
def mock_ocr_client():
    """Provide a mock Google Cloud Vision client for OCR testing"""
    with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client:
        # Create mock response structure
        mock_response = Mock()
        mock_page = Mock()
        mock_block = Mock()
        mock_block.confidence = 0.95
        mock_block.text = "Sample mathematical text with LaTeX: $x^2 + y^2 = r^2$"
        mock_block.bounding_box.vertices = [
            Mock(x=0, y=0),
            Mock(x=100, y=0),
            Mock(x=100, y=100),
            Mock(x=0, y=100)
        ]
        mock_page.blocks = [mock_block]
        mock_response.pages = [mock_page]
        mock_response.full_text_annotation.text = mock_block.text
        
        mock_client.return_value.document_text_detection.return_value = mock_response
        yield mock_client

@pytest.fixture
def ocr_processor(mock_ocr_client):
    """Provide configured OCR processor for testing"""
    return OCRProcessor("test_api_key")

@pytest.fixture
def test_pdf_with_latex(test_storage_path):
    """Create a test PDF with LaTeX content"""
    pdf_path = test_storage_path / "test_latex.pdf"
    doc = fitz.open()
    page = doc.new_page()
    
    latex_content = r"""
    \begin{theorem}[Pythagorean Theorem]
    In a right triangle with sides a, b, and hypotenuse c:
    $a^2 + b^2 = c^2$
    \end{theorem}
    
    \begin{proof}
    Consider the square constructed on the hypotenuse...
    \end{proof}
    """
    
    page.insert_text((50, 50), latex_content)
    doc.save(pdf_path)
    doc.close()
    
    return pdf_path