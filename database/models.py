"""
Database Models Module

Defines SQLAlchemy ORM models for storing theorem-proof pairs and their metadata,
with support for vector embeddings using pgvector.
"""

from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table, JSON, Enum
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
import enum
from typing import List

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

class ProofTechniqueEnum(enum.Enum):
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CONSTRUCTIVE = "constructive"
    PROBABILISTIC = "probabilistic"
    COMBINATORIAL = "combinatorial"
    ALGEBRAIC = "algebraic"
    GEOMETRIC = "geometric"

class Concept(Base):
    __tablename__ = 'concepts'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String)
    parent_concepts = Column(ARRAY(String))
    related_concepts = Column(ARRAY(String))
    embedding = Column(Vector(384))  # Dimension matches the sentence transformer model

class Theorem(Base):
    __tablename__ = 'theorems'

    id = Column(String, primary_key=True)
    source_id = Column(String, ForeignKey('sources.id'))
    theorem_type = Column(String, nullable=False)
    theorem_number = Column(String)
    theorem_title = Column(String)
    theorem_text = Column(String, nullable=False)
    proof_text = Column(String, nullable=False)
    page_number = Column(Integer)
    
    # Metadata
    proof_techniques = Column(ARRAY(Enum(ProofTechniqueEnum)))
    difficulty_score = Column(Float)
    proof_length = Column(Integer)
    confidence_score = Column(Float)
    
    # Rich content
    spatial_info = Column(JSON)
    formatting = Column(JSON)
    
    # Vector embeddings for similarity search
    theorem_embedding = Column(Vector(384))
    proof_embedding = Column(Vector(384))
    
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

class Source(Base):
    __tablename__ = 'sources'

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    authors = Column(ARRAY(String))
    year = Column(Integer)
    source_type = Column(String)  # e.g., 'paper', 'book', 'thesis'
    license = Column(String)
    
    theorems = relationship("Theorem", back_populates="source")

class GraphEdge(Base):
    __tablename__ = 'graph_edges'

    id = Column(Integer, primary_key=True)
    from_theorem_id = Column(String, ForeignKey('theorems.id'))
    to_theorem_id = Column(String, ForeignKey('theorems.id'))
    edge_type = Column(String)  # logical_dependency, proof_technique, equivalence
    weight = Column(Float)
    metadata = Column(JSON)