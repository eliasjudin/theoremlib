"""
FastAPI Main Module

Provides RESTful API endpoints for the theorem database system,
handling ingestion, querying, and graph operations.
"""

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
import json

from ..database.database_loader import DatabaseLoader
from ..database.models import Theorem, Source
from ..ingest.ingest import PDFIngestor, PDFMetadata
from ..extraction.content_extraction import ContentExtractor, TheoremProofPair
from ..annotation.metadata_annotation import MetadataAnnotator, TheoremAnnotation
from ..graph.graph_models import TheoremGraph, EdgeType, GraphEdgeMetadata

app = FastAPI(
    title="TheoremLib API",
    description="API for managing mathematical theorems and proofs",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
db_loader = DatabaseLoader("postgresql+asyncpg://user:password@localhost/theoremlib")

# Initialize components
pdf_ingestor = PDFIngestor("storage/pdfs")
content_extractor = ContentExtractor()
metadata_annotator = MetadataAnnotator("ontology/math_ontology.json")
theorem_graph = TheoremGraph()

# Dependency for database sessions
async def get_db():
    async for session in db_loader.get_db():
        yield session

# API Models
class TheoremResponse(BaseModel):
    id: str
    theorem_type: str
    theorem_text: str
    proof_text: str
    metadata: dict

class GraphResponse(BaseModel):
    nodes: List[dict]
    edges: List[dict]

# Endpoints
@app.post("/ingest/pdf", response_model=List[str])
async def ingest_pdf(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Ingest a PDF document and extract theorems"""
    try:
        # Process PDF
        pdf_metadata = await pdf_ingestor.ingest_pdf(file.file, {})
        
        # Extract theorems
        theorem_pairs = content_extractor.extract_pairs(
            pdf_metadata.text_content,
            pdf_metadata.spatial_info,
            pdf_metadata.page_number
        )
        
        # Annotate and store theorems
        theorem_ids = []
        for pair in theorem_pairs:
            annotation = await metadata_annotator.annotate_theorem(
                pair.theorem_text,
                pair.proof_text
            )
            
            theorem = Theorem(
                theorem_text=pair.theorem_text,
                proof_text=pair.proof_text,
                **annotation.dict()
            )
            
            theorem_id = await db_loader.store_theorem(db, theorem)
            theorem_ids.append(theorem_id)
            
            # Update theorem graph
            theorem_graph.add_theorem_node(theorem_id, annotation.dict())
            
        return theorem_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/theorems/{theorem_id}", response_model=TheoremResponse)
async def get_theorem(
    theorem_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific theorem by ID"""
    theorem = await db.get(Theorem, theorem_id)
    if not theorem:
        raise HTTPException(status_code=404, detail="Theorem not found")
    return TheoremResponse(
        id=theorem.id,
        theorem_type=theorem.theorem_type,
        theorem_text=theorem.theorem_text,
        proof_text=theorem.proof_text,
        metadata={
            "proof_techniques": theorem.proof_techniques,
            "difficulty_score": theorem.difficulty_score,
            "concepts": [c.name for c in theorem.concepts],
            "prerequisites": [p.id for p in theorem.prerequisites]
        }
    )

@app.get("/theorems/search", response_model=List[TheoremResponse])
async def search_theorems(
    query: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Search theorems using full-text search"""
    theorems = await db_loader.full_text_search(db, query, limit)
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
        for t in theorems
    ]

@app.get("/graph/dependencies/{theorem_id}", response_model=GraphResponse)
async def get_theorem_dependencies(
    theorem_id: str,
    depth: int = 2,
    db: AsyncSession = Depends(get_db)
):
    """Get the dependency graph for a theorem"""
    subgraph = theorem_graph.get_dependencies(theorem_id, depth)
    return GraphResponse(
        nodes=[
            {"id": n, **subgraph.nodes[n]}
            for n in subgraph.nodes()
        ],
        edges=[
            {
                "from": u,
                "to": v,
                "type": k,
                "metadata": theorem_graph.edge_metadata.get(
                    (u, v, k),
                    GraphEdgeMetadata(confidence=0.5)
                ).__dict__
            }
            for u, v, k in subgraph.edges(keys=True)
        ]
    )

@app.get("/graph/similar/{theorem_id}", response_model=List[TheoremResponse])
async def find_similar_theorems(
    theorem_id: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Find theorems with similar proof techniques"""
    theorem = await db.get(Theorem, theorem_id)
    if not theorem:
        raise HTTPException(status_code=404, detail="Theorem not found")
        
    similar = await db_loader.vector_search(
        db,
        theorem.theorem_embedding,
        limit
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
        for t in similar
    ]