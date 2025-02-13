"""
FastAPI Main Module

Provides RESTful API endpoints for the theorem database system,
handling ingestion, querying, and graph operations.
"""

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict
from pydantic import BaseModel
import tempfile
import shutil
import json
import os
from datetime import datetime
import logging

from ..database.database_loader import DatabaseLoader
from ..database.models import Theorem, Source, ProcessingStatus
from ..ingest.ingest import PDFIngestor, DocumentMetadata
from ..main_pipeline import PipelineCoordinator
from ..extraction.content_extraction import ContentExtractor
from ..annotation.metadata_annotation import MetadataAnnotator
from ..graph.graph_models import TheoremGraph, EdgeType, GraphEdgeMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Models
class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    queued_time: datetime

class TheoremResponse(BaseModel):
    id: str
    theorem_type: str
    theorem_text: str
    proof_text: str
    metadata: dict

class ProcessingStatusResponse(BaseModel):
    document_id: str
    status: ProcessingStatus
    error_message: Optional[str]
    completed_steps: List[str]
    total_theorems: Optional[int]
    processed_theorems: Optional[int]

# Initialize FastAPI app with OpenAPI metadata
app = FastAPI(
    title="TheoremLib API",
    description="API for managing mathematical theorems and proofs",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(os.getenv("ALLOWED_ORIGINS", '["*"]')),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components from environment variables
db_loader = DatabaseLoader(os.getenv("DATABASE_URL"))
pipeline_coordinator = PipelineCoordinator(
    db_url=os.getenv("DATABASE_URL"),
    storage_path=os.getenv("STORAGE_PATH", "storage"),
    ontology_path=os.getenv("ONTOLOGY_PATH", "ontology/math_ontology.json"),
    ocr_api_key=os.getenv("OCR_API_KEY", "")
)

# Dependency for database sessions
async def get_db():
    async for session in db_loader.get_db():
        yield session

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source: str = Form(...),
    title: str = Form(...),
    author: str = Form(...),
    page_count: int = Form(...),
    theorem_number: Optional[str] = Form(""),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a PDF document with metadata for processing.
    
    The document will be processed asynchronously in the background.
    Returns a document ID that can be used to check processing status.
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        # Create metadata
        metadata = DocumentMetadata(
            source=source,
            title=title,
            author=author,
            page_count=page_count,
            theorem_number=theorem_number,
            upload_time=datetime.utcnow()
        )

        # Queue document for processing
        doc_id = await db_loader.create_processing_status(
            db,
            file_path=tmp_path,
            metadata=metadata
        )

        # Start background processing
        background_tasks.add_task(
            pipeline_coordinator.process_document,
            tmp_path,
            metadata
        )

        return DocumentUploadResponse(
            document_id=doc_id,
            status="queued",
            queued_time=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{document_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the current processing status of a document.
    """
    status = await db_loader.get_processing_status(db, document_id)
    if not status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return ProcessingStatusResponse(
        document_id=document_id,
        status=status.status,
        error_message=status.error_message,
        completed_steps=status.completed_steps,
        total_theorems=status.total_theorems,
        processed_theorems=status.processed_theorems
    )

@app.post("/reprocess/{document_id}")
async def trigger_reprocessing(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Manually trigger reprocessing of a document.
    """
    status = await db_loader.get_processing_status(db, document_id)
    if not status:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if status.status == ProcessingStatus.PROCESSING:
        raise HTTPException(
            status_code=400,
            detail="Document is currently being processed"
        )

    # Reset processing status
    await db_loader.update_processing_status(
        db,
        document_id,
        ProcessingStatus.QUEUED
    )

    # Queue reprocessing
    background_tasks.add_task(
        pipeline_coordinator.process_document,
        status.file_path,
        status.metadata
    )

    return {"message": "Reprocessing queued"}

@app.get("/query", response_model=List[TheoremResponse])
async def query_theorems(
    query: str,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for theorems using full-text search and vector similarity.
    """
    results = await db_loader.full_text_search(
        db,
        query,
        limit=limit,
        offset=offset
    )
    
    return [
        TheoremResponse(
            id=t.id,
            theorem_type=t.theorem_type,
            theorem_text=t.theorem_text,
            proof_text=t.proof_text,
            metadata={
                "proof_techniques": t.proof_techniques,
                "difficulty_score": t.difficulty_score,
                "concepts": [c.name for c in t.concepts],
                "prerequisites": [p.id for p in t.prerequisites]
            }
        )
        for t in results
    ]

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Check API health status
    """
    return {"status": "healthy"}