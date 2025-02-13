"""
Database Models Module

Defines SQLAlchemy ORM models for storing theorem-proof pairs and their metadata,
with support for vector embeddings using pgvector.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, JSON, ForeignKey, 
    Table, Float, Index, Enum, func
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
import enum
import uuid
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Vector type handling
try:
    from pgvector.sqlalchemy import Vector as PGVector
    VECTOR_TYPE = PGVector
    HAVE_PGVECTOR = True
    logger.info("Using pgvector for vector operations")
except ImportError:
    # Fallback to regular array type
    VECTOR_TYPE = lambda dim: ARRAY(Float)
    HAVE_PGVECTOR = False
    logger.warning("pgvector not available, falling back to array type")

Base = declarative_base()

# Association tables
theorem_concepts = Table(
    'theorem_concepts', 
    Base.metadata,
    Column('theorem_id', String, ForeignKey('theorems.id')),
    Column('concept_id', String, ForeignKey('concepts.id'))
)

theorem_prerequisites = Table(
    'theorem_prerequisites',
    Base.metadata,
    Column('theorem_id', String, ForeignKey('theorems.id')),
    Column('prerequisite_id', String, ForeignKey('theorems.id'))
)

class ProofTechniqueEnum(str, enum.Enum):
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CONSTRUCTIVE = "constructive"
    PROBABILISTIC = "probabilistic"
    COMBINATORIAL = "combinatorial"
    ALGEBRAIC = "algebraic"
    GEOMETRIC = "geometric"

class Source(Base):
    """Model for storing source document metadata"""
    __tablename__ = 'sources'

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    authors = Column(ARRAY(String))
    year = Column(Integer)
    source_type = Column(String)
    license = Column(String)
    
    theorems = relationship("Theorem", back_populates="source")
    
    __table_args__ = (
        Index('idx_sources_title_authors', 'title', postgresql_using='gin'),
    )

class Concept(Base):
    """Model for storing mathematical concepts from the ontology"""
    __tablename__ = 'concepts'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String)
    parent_concepts = Column(ARRAY(String))
    related_concepts = Column(ARRAY(String))
    embedding = Column(VECTOR_TYPE(384))  # For semantic similarity search

    __table_args__ = (
        Index('idx_concept_name_trgm', 'name', postgresql_using='gin', 
              postgresql_ops={'name': 'gin_trgm_ops'}),
    )

    if HAVE_PGVECTOR:
        __table_args__ = (
            *__table_args__,
            Index('idx_concept_embedding', 'embedding', postgresql_using='ivfflat',
                  postgresql_with={'lists': 100})
        )

    def compute_similarity(self, other_embedding: List[float]) -> float:
        """Compute similarity with another embedding"""
        if not self.embedding or not other_embedding:
            return 0.0
        try:
            if HAVE_PGVECTOR:
                # Use L2 distance for pgvector
                vec1 = np.array(self.embedding)
                vec2 = np.array(other_embedding)
                return 1.0 - np.linalg.norm(vec1 - vec2)
            else:
                # Fallback to cosine similarity
                vec1 = np.array(self.embedding)
                vec2 = np.array(other_embedding)
                return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception:
            return 0.0

class Theorem(Base):
    """Model for storing theorem-proof pairs with rich metadata"""
    __tablename__ = 'theorems'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String, ForeignKey('sources.id'))
    theorem_type = Column(String, nullable=False)
    theorem_number = Column(String)
    theorem_title = Column(String)
    theorem_text = Column(Text, nullable=False)
    proof_text = Column(Text, nullable=False)
    page_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Metadata fields
    proof_techniques = Column(ARRAY(Enum(ProofTechniqueEnum)))
    difficulty_score = Column(Float)
    proof_length = Column(Integer)
    confidence_score = Column(Float)
    
    # Rich content
    spatial_info = Column(JSON)
    formatting = Column(JSON)
    ontology_mapping = Column(JSON)
    
    # Vector embeddings for similarity search
    theorem_embedding = Column(VECTOR_TYPE(384))
    proof_embedding = Column(VECTOR_TYPE(384))
    
    # Relationships
    source = relationship("Source", back_populates="theorems")
    concepts = relationship("Concept", secondary=theorem_concepts)
    prerequisites = relationship(
        "Theorem",
        secondary=theorem_prerequisites,
        primaryjoin=id==theorem_prerequisites.c.theorem_id,
        secondaryjoin=id==theorem_prerequisites.c.prerequisite_id,
        backref="dependent_theorems"
    )
    
    __table_args__ = (
        Index('idx_theorem_text_gin', 
              func.to_tsvector('english', theorem_text),
              postgresql_using='gin'),
        Index('idx_proof_text_gin',
              func.to_tsvector('english', proof_text),
              postgresql_using='gin'),
    )

    if HAVE_PGVECTOR:
        __table_args__ = (
            *__table_args__,
            Index('idx_theorem_embedding',
                  'theorem_embedding',
                  postgresql_using='ivfflat',
                  postgresql_with={'lists': 100}),
            Index('idx_proof_embedding',
                  'proof_embedding',
                  postgresql_using='ivfflat',
                  postgresql_with={'lists': 100})
        )

    def compute_similarity(self, other_theorem: 'Theorem') -> float:
        """Compute similarity with another theorem"""
        if not self.theorem_embedding or not other_theorem.theorem_embedding:
            return 0.0
        try:
            if HAVE_PGVECTOR:
                # Use L2 distance for pgvector
                vec1 = np.array(self.theorem_embedding)
                vec2 = np.array(other_theorem.theorem_embedding)
                return 1.0 - np.linalg.norm(vec1 - vec2)
            else:
                # Fallback to cosine similarity
                vec1 = np.array(self.theorem_embedding)
                vec2 = np.array(other_theorem.theorem_embedding)
                return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception:
            return 0.0

class TheoremCreate(BaseModel):
    """Pydantic model for creating a new theorem"""
    theorem_type: str = Field(..., description="Type of theorem (theorem, lemma, etc.)")
    theorem_number: Optional[str] = Field(None, description="Original theorem number")
    theorem_title: Optional[str] = Field(None, description="Title or name of theorem")
    theorem_text: str = Field(..., description="Full text of theorem statement")
    proof_text: str = Field(..., description="Full text of proof")
    page_number: Optional[int] = Field(None, description="Page number in source")
    source_id: str = Field(..., description="ID of source document")
    proof_techniques: List[ProofTechniqueEnum] = Field(..., description="List of proof techniques used")
    difficulty_score: float = Field(..., ge=0.0, le=1.0, description="Computed difficulty score")
    concepts: List[str] = Field(..., description="List of concept IDs")
    spatial_info: Dict[str, Any] = Field(default_factory=dict, description="Spatial layout information")
    formatting: Dict[str, Any] = Field(default_factory=dict, description="Formatting metadata")
    ontology_mapping: Dict[str, Any] = Field(..., description="Mapping to math ontology")
    theorem_embedding: Optional[List[float]] = None
    proof_embedding: Optional[List[float]] = None

    @validator('difficulty_score')
    def validate_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Difficulty score must be between 0 and 1')
        return v

class TheoremResponse(BaseModel):
    """Pydantic model for theorem responses"""
    id: str
    theorem_type: str
    theorem_number: Optional[str]
    theorem_title: Optional[str]
    theorem_text: str
    proof_text: str
    page_number: Optional[int]
    source: Dict[str, Any]
    proof_techniques: List[ProofTechniqueEnum]
    difficulty_score: float
    concepts: List[Dict[str, Any]]
    created_at: datetime
    ontology_mapping: Dict[str, Any]
    
    class Config:
        orm_mode = True

class GraphEdge(Base):
    """Model for storing theorem relationships"""
    __tablename__ = 'graph_edges'

    id = Column(Integer, primary_key=True)
    from_theorem_id = Column(String, ForeignKey('theorems.id'))
    to_theorem_id = Column(String, ForeignKey('theorems.id'))
    edge_type = Column(String)
    weight = Column(Float)
    metadata = Column(JSON)