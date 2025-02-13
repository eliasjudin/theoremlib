"""
Tests for theorem extraction and annotation functionality
"""

import pytest
from pathlib import Path
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import hashlib

from ..extraction.content_extraction import TheoremProofPair, TheoremType
from ..annotation.metadata_annotation import TheoremAnnotation
from ..database.models import Theorem, Concept
from ..graph.graph_models import TheoremGraph, EdgeType, GraphEdgeMetadata
from ..ingest.ingest import DocumentMetadata, PDFMetadata, PDFIngestor

@pytest.mark.asyncio
async def test_pdf_ingestion(pdf_ingestor: PDFIngestor, test_pdf_path: Path):
    """Test PDF ingestion and text layer validation"""
    metadata = await pdf_ingestor.ingest_pdf(test_pdf_path, {
        "title": "Test Theorem Paper",
        "authors": ["Test Author"],
        "year": 2024,
    })
    
    assert metadata.has_text_layer
    assert not metadata.needs_ocr
    assert metadata.page_count > 0

@pytest.mark.asyncio
async def test_theorem_extraction(content_extractor):
    """Test theorem-proof pair extraction"""
    test_text = """
    Theorem 1.1 (Fundamental Theorem). For all x ∈ R, x² ≥ 0.
    
    Proof. Let x be any real number. Then x² is the product of x with itself.
    By the properties of real numbers, this product is always non-negative.
    Therefore, x² ≥ 0. □
    """
    
    pairs = content_extractor.extract_pairs(test_text, {}, 1)
    assert len(pairs) == 1
    assert pairs[0].theorem_type == TheoremType.THEOREM
    assert pairs[0].theorem_number == "1.1"
    assert "x² ≥ 0" in pairs[0].theorem_text
    assert "□" in pairs[0].proof_text

@pytest.mark.asyncio
async def test_metadata_annotation(metadata_annotator):
    """Test theorem metadata annotation"""
    theorem_text = "Theorem: Every continuous function on a closed interval is bounded."
    proof_text = """Proof by contradiction. Assume f is continuous on [a,b] but unbounded.
    Then for each n ∈ N, there exists x_n ∈ [a,b] with |f(x_n)| > n.
    By Bolzano-Weierstrass, {x_n} has a convergent subsequence. This leads to a contradiction
    with the continuity of f. Therefore, f must be bounded. □"""
    
    annotation = await metadata_annotator.annotate_theorem(theorem_text, proof_text)
    
    assert isinstance(annotation, TheoremAnnotation)
    assert "contradiction" in [t.value for t in annotation.proof_techniques]
    assert annotation.difficulty_score > 0
    assert len(annotation.key_concepts) > 0

@pytest.mark.asyncio
async def test_database_storage(test_db):
    """Test database storage and retrieval"""
    async with test_db() as session:
        # Create test concept
        concept = Concept(
            id="test.continuity",
            name="Continuity",
            category="analysis"
        )
        session.add(concept)
        
        # Create test theorem
        theorem = Theorem(
            id="test-theorem-1",
            theorem_type="theorem",
            theorem_text="Test theorem about continuity",
            proof_text="Test proof using contradiction",
            proof_techniques=["contradiction"],
            difficulty_score=0.7
        )
        theorem.concepts.append(concept)
        
        session.add(theorem)
        await session.commit()
        
        # Verify storage
        stored_theorem = await session.get(Theorem, "test-theorem-1")
        assert stored_theorem is not None
        assert stored_theorem.theorem_text == "Test theorem about continuity"
        assert len(stored_theorem.concepts) == 1
        assert stored_theorem.concepts[0].name == "Continuity"

@pytest.mark.asyncio
async def test_graph_operations():
    """Test theorem graph operations"""
    graph = TheoremGraph()
    
    # Add test theorems
    graph.add_theorem_node("theorem1", {"title": "First Theorem"})
    graph.add_theorem_node("theorem2", {"title": "Second Theorem"})
    
    # Add dependency
    graph.add_edge(
        "theorem2",
        "theorem1",
        EdgeType.LOGICAL_DEPENDENCY,
        GraphEdgeMetadata(confidence=0.9, description="Uses theorem 1")
    )
    
    # Test graph operations
    deps = graph.get_dependencies("theorem2")
    assert len(deps.nodes) == 2
    assert len(deps.edges) == 1
    
    impact = graph.calculate_theorem_impact("theorem1")
    assert impact > 0  # Should have impact since theorem2 depends on it

@pytest.mark.asyncio
async def test_document_metadata():
    """Test document metadata validation and handling"""
    # Test basic metadata creation
    doc_metadata = DocumentMetadata(
        source="arXiv",
        title="Test Mathematics Paper",
        author="Test Author",
        page_count=10,
        theorem_number="1.1"
    )
    
    assert doc_metadata.source == "arXiv"
    assert doc_metadata.upload_time.tzinfo == timezone.utc
    
    # Test metadata is preserved in PDF ingestion
    metadata = await pdf_ingestor.ingest_pdf(test_pdf_path, {
        "title": "Test Theorem Paper",
        "authors": ["Test Author"],
        "year": 2024,
    }, doc_metadata=doc_metadata)
    
    assert metadata.document_metadata is not None
    assert metadata.document_metadata.title == "Test Mathematics Paper"
    assert metadata.document_metadata.author == "Test Author"

@pytest.mark.asyncio
async def test_pdf_validation_and_deduplication(pdf_ingestor, tmp_path):
    """Test PDF validation and duplicate detection"""
    # Create a test PDF file
    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"%PDF-1.4\n%EOF")  # Minimal valid PDF
    
    # Test successful ingestion
    doc_metadata = DocumentMetadata(
        source="test",
        title="Test PDF",
        author="Test Author",
        page_count=1
    )
    
    metadata = await pdf_ingestor.ingest_pdf(test_file, {}, doc_metadata)
    assert metadata.document_metadata.file_hash is not None
    
    # Test duplicate detection
    with pytest.raises(ValueError, match="already been processed"):
        await pdf_ingestor.ingest_pdf(test_file, {}, doc_metadata)
    
@pytest.mark.asyncio
async def test_invalid_pdf_handling(pdf_ingestor, tmp_path):
    """Test handling of invalid PDF files"""
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        await pdf_ingestor.ingest_pdf(
            tmp_path / "nonexistent.pdf",
            {},
            None
        )
    
    # Test invalid file type
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("Not a PDF")
    with pytest.raises(ValueError, match="Invalid file type"):
        await pdf_ingestor.ingest_pdf(invalid_file, {}, None)
    
    # Test corrupted PDF
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_text("corrupted content")
    with pytest.raises(ValueError, match="Invalid or corrupted PDF"):
        await pdf_ingestor.ingest_pdf(corrupt_pdf, {}, None)

@pytest.mark.asyncio
async def test_metadata_validation():
    """Test metadata validation rules"""
    # Test invalid page count
    with pytest.raises(ValueError, match="Page count must be positive"):
        DocumentMetadata(
            source="test",
            title="Test",
            author="Author",
            page_count=0
        )
    
    # Test empty title
    with pytest.raises(ValueError, match="Title cannot be empty"):
        DocumentMetadata(
            source="test",
            title="   ",
            author="Author",
            page_count=1
        )
    
    # Test valid metadata with file hash
    metadata = DocumentMetadata(
        source="test",
        title="Valid Title",
        author="Author",
        page_count=1,
        file_hash="abc123"
    )
    assert metadata.file_hash == "abc123"
    assert metadata.upload_time.tzinfo == timezone.utc