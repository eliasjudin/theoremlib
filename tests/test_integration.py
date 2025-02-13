"""
Tests for concurrent PDF ingestion and thread safety
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
import fitz

from ..ingest.ingest import DocumentMetadata, PDFIngestor, PDFMetadata
from ..ocr.ocr_module import OCRProcessor, OCRResult, ExtractionMethod

@pytest.mark.asyncio
async def test_concurrent_pdf_ingestion(pdf_ingestor, tmp_path):
    """Test concurrent PDF uploads with duplicate detection"""
    # Create test PDFs
    test_files = []
    for i in range(3):
        pdf_path = tmp_path / f"test_{i}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\ntest content\n%EOF")
        test_files.append(pdf_path)

    # Create metadata for each file
    metadatas = [
        DocumentMetadata(
            source="test",
            title=f"Test PDF {i}",
            author="Test Author",
            page_count=1
        )
        for i in range(3)
    ]

    # Try to ingest the same file concurrently
    async def ingest_attempt(file_path, metadata):
        try:
            return await pdf_ingestor.ingest_pdf(file_path, {}, metadata)
        except ValueError as e:
            if "already been processed" in str(e):
                return "duplicate"
            raise

    # Test concurrent ingestion of different files
    results = await asyncio.gather(
        *[ingest_attempt(f, m) for f, m in zip(test_files, metadatas)]
    )
    
    # Verify results
    successful = [r for r in results if r != "duplicate"]
    assert len(successful) == len(test_files)  # All unique files should succeed

    # Test concurrent ingestion of the same file
    duplicate_results = await asyncio.gather(
        *[ingest_attempt(test_files[0], metadatas[0]) for _ in range(3)]
    )
    
    # Verify only one succeeded
    successful_duplicates = [r for r in duplicate_results if r != "duplicate"]
    assert len(successful_duplicates) == 1

@pytest.mark.asyncio
async def test_failed_ingestion_cleanup(pdf_ingestor, tmp_path):
    """Test cleanup of partially processed files on failure"""
    # Create a PDF that will fail validation halfway through
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\n" + b"0" * 1024)  # Valid header but corrupt content

    initial_files = set(pdf_ingestor.ingested_files_path.glob("*"))
    
    with pytest.raises(ValueError):
        await pdf_ingestor.ingest_pdf(
            corrupt_pdf,
            {},
            DocumentMetadata(
                source="test",
                title="Corrupt PDF",
                author="Test Author",
                page_count=1
            )
        )

    # Verify no files were left behind
    final_files = set(pdf_ingestor.ingested_files_path.glob("*"))
    assert initial_files == final_files  # No new files should remain

@pytest.mark.asyncio
async def test_hash_persistence(pdf_ingestor, tmp_path):
    """Test that processed file hashes persist across instances"""
    # Create and ingest a test PDF
    test_file = tmp_path / "persistence_test.pdf"
    test_file.write_bytes(b"%PDF-1.4\ntest content\n%EOF")
    
    metadata = DocumentMetadata(
        source="test",
        title="Test PDF",
        author="Test Author",
        page_count=1
    )
    
    await pdf_ingestor.ingest_pdf(test_file, {}, metadata)
    
    # Create new ingestor instance
    new_ingestor = PDFIngestor(pdf_ingestor.storage_path)
    
    # Try to ingest the same file
    with pytest.raises(ValueError, match="already been processed"):
        await new_ingestor.ingest_pdf(test_file, {}, metadata)

@pytest.mark.asyncio
async def test_latex_ocr_integration(test_pdf_with_latex, ocr_processor, pdf_ingestor):
    """Test integration between OCR and PDF ingestion with LaTeX content"""
    # First verify the LaTeX layer
    has_latex, confidence = ocr_processor.has_complete_latex_layer(test_pdf_with_latex)
    assert has_latex
    assert confidence > 0.9

    # Process with OCR module
    ocr_results = await ocr_processor.process_pdf(test_pdf_with_latex)
    assert len(ocr_results) > 0
    assert ocr_results[0].extraction_method == ExtractionMethod.LATEX
    assert "theorem" in ocr_results[0].text_content.lower()
    assert "proof" in ocr_results[0].text_content.lower()

    # Now test full ingestion pipeline
    doc_metadata = DocumentMetadata(
        source="arXiv",
        title="LaTeX Test Paper",
        author="Test Author",
        page_count=1,
        theorem_number="1"
    )
    
    pdf_metadata = await pdf_ingestor.ingest_pdf(test_pdf_with_latex, {}, doc_metadata)
    assert pdf_metadata.has_text_layer
    assert not pdf_metadata.needs_ocr

@pytest.mark.asyncio
async def test_ocr_fallback_pipeline(tmp_path, ocr_processor, pdf_ingestor):
    """Test full pipeline with OCR fallback for scanned documents"""
    # Create a PDF without LaTeX layer (just images)
    scanned_pdf = tmp_path / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Add some dummy image content
    page.insert_text((50, 50), "Theorem 1: Sample theorem text")
    doc.save(scanned_pdf)
    doc.close()

    # Verify it needs OCR
    has_latex, confidence = ocr_processor.has_complete_latex_layer(scanned_pdf)
    assert not has_latex or confidence < 0.9

    # Test OCR processing
    ocr_results = await ocr_processor.process_pdf(scanned_pdf)
    assert len(ocr_results) > 0
    assert ocr_results[0].extraction_method == ExtractionMethod.OCR

    # Test full ingestion pipeline
    doc_metadata = DocumentMetadata(
        source="scanned",
        title="Scanned Test Paper",
        author="Test Author",
        page_count=1
    )
    
    pdf_metadata = await pdf_ingestor.ingest_pdf(scanned_pdf, {}, doc_metadata)
    assert not pdf_metadata.has_text_layer
    assert pdf_metadata.needs_ocr

@pytest.mark.asyncio
async def test_mixed_content_handling(tmp_path, ocr_processor):
    """Test handling of PDFs with mixed LaTeX and scanned content"""
    # Create a PDF with mixed content
    mixed_pdf = tmp_path / "mixed.pdf"
    doc = fitz.open()
    
    # Page 1: LaTeX content
    page1 = doc.new_page()
    page1.insert_text((50, 50), r"""
    \begin{theorem}
    LaTeX content here
    \end{theorem}
    """)
    
    # Page 2: Image-based content
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Plain text without LaTeX markers")
    
    doc.save(mixed_pdf)
    doc.close()

    # Process the mixed content PDF
    results = await ocr_processor.process_pdf(mixed_pdf)
    
    # Verify appropriate handling of different pages
    assert len(results) == 2
    assert any(r.extraction_method == ExtractionMethod.LATEX for r in results)
    
    # Check confidence scores
    confidences = [r.confidence for r in results]
    assert max(confidences) > 0.9  # At least one page should have high confidence