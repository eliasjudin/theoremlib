"""
Integration tests for the complete theorem processing pipeline
"""

import pytest
from pathlib import Path
import tempfile
import os
import json
import asyncio
from datetime import datetime

from ..main_pipeline import PipelineCoordinator
from ..database.database_loader import DatabaseLoader
from ..ingest.ingest import DocumentMetadata
from ..database.models import ProcessingStatus
from .test_core import sample_metadata, SAMPLE_THEOREM, SAMPLE_PROOF

@pytest.fixture
async def pipeline_coordinator(db_session, temp_ontology):
    """Create a pipeline coordinator with test configuration"""
    with tempfile.TemporaryDirectory() as storage:
        coordinator = PipelineCoordinator(
            db_url=os.getenv("TEST_DATABASE_URL"),
            storage_path=storage,
            ontology_path=str(temp_ontology),
            ocr_api_key="test_key"
        )
        yield coordinator

@pytest.mark.asyncio
async def test_complete_pipeline(pipeline_coordinator, sample_metadata):
    """Test complete pipeline from PDF ingestion to graph construction"""
    # Create test PDF with theorem
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.7\n" + SAMPLE_THEOREM.encode() + b"\n" + SAMPLE_PROOF.encode())
        pdf_path = tmp.name

    try:
        # Process document
        metadata = DocumentMetadata(**sample_metadata)
        doc_id = await pipeline_coordinator.process_document(pdf_path, metadata)
        
        # Check processing status
        status = await pipeline_coordinator.get_processing_status(doc_id)
        assert status.status == ProcessingStatus.COMPLETED
        
        # Verify theorem extraction
        theorems = await pipeline_coordinator.get_extracted_theorems(doc_id)
        assert len(theorems) > 0
        assert theorems[0].theorem_text is not None
        
        # Check metadata annotation
        assert theorems[0].proof_techniques is not None
        assert isinstance(theorems[0].difficulty_score, float)
        
        # Verify graph construction
        graph = await pipeline_coordinator.get_theorem_graph(theorems[0].id)
        assert graph is not None
    
    finally:
        os.unlink(pdf_path)

@pytest.mark.asyncio
async def test_error_recovery(pipeline_coordinator, sample_metadata):
    """Test error recovery and retries in pipeline"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Create corrupted PDF
        tmp.write(b"Not a valid PDF")
        pdf_path = tmp.name

    try:
        metadata = DocumentMetadata(**sample_metadata)
        with pytest.raises(Exception):
            await pipeline_coordinator.process_document(pdf_path, metadata)
            
        # Check error status
        status = await pipeline_coordinator.get_processing_status(metadata.source)
        assert status.status == ProcessingStatus.ERROR
        assert status.error_message is not None
        
    finally:
        os.unlink(pdf_path)

@pytest.mark.asyncio
async def test_concurrent_processing(pipeline_coordinator):
    """Test concurrent document processing"""
    pdf_files = []
    metadatas = []
    
    # Create multiple test PDFs
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.7\n" + f"Theorem {i}".encode())
            pdf_files.append(tmp.name)
            metadatas.append(DocumentMetadata(
                source=f"Source{i}",
                title=f"Title{i}",
                author="Test Author",
                page_count=1
            ))
    
    try:
        # Process documents concurrently
        tasks = [
            pipeline_coordinator.process_document(pdf, meta)
            for pdf, meta in zip(pdf_files, metadatas)
        ]
        doc_ids = await asyncio.gather(*tasks)
        
        # Verify all documents processed
        for doc_id in doc_ids:
            status = await pipeline_coordinator.get_processing_status(doc_id)
            assert status.status in [ProcessingStatus.COMPLETED, ProcessingStatus.ERROR]
            
    finally:
        for pdf in pdf_files:
            os.unlink(pdf)

@pytest.mark.asyncio
async def test_reprocessing(pipeline_coordinator, sample_metadata):
    """Test manual document reprocessing"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.7\n" + SAMPLE_THEOREM.encode())
        pdf_path = tmp.name

    try:
        metadata = DocumentMetadata(**sample_metadata)
        doc_id = await pipeline_coordinator.process_document(pdf_path, metadata)
        
        # Trigger reprocessing
        await pipeline_coordinator.reprocess_document(doc_id)
        
        # Verify reprocessing status
        status = await pipeline_coordinator.get_processing_status(doc_id)
        assert status.status == ProcessingStatus.COMPLETED
        
    finally:
        os.unlink(pdf_path)

@pytest.mark.asyncio
async def test_api_integration(pipeline_coordinator, sample_metadata):
    """Test integration with FastAPI endpoints"""
    from fastapi.testclient import TestClient
    from ..api.main import app
    
    client = TestClient(app)
    
    # Test file upload
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(b"%PDF-1.7\n" + SAMPLE_THEOREM.encode())
        response = client.post(
            "/upload",
            files={"file": tmp},
            data=sample_metadata
        )
        assert response.status_code == 200
        doc_id = response.json()["document_id"]
        
        # Check processing status
        response = client.get(f"/status/{doc_id}")
        assert response.status_code == 200
        
        # Test theorem query
        response = client.get("/query", params={"query": "Pythagorean"})
        assert response.status_code == 200