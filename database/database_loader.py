"""
Database Loader Module

Handles database operations, connection management, and migrations using SQLAlchemy and Alembic.
"""

from sqlalchemy import create_engine, text, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, AsyncGenerator
import logging
from pathlib import Path

from .models import (
    Base, Theorem, Concept, Source, GraphEdge, 
    TheoremCreate, TheoremResponse, HAVE_PGVECTOR
)

logger = logging.getLogger(__name__)

class DatabaseLoader:
    def __init__(self, database_url: str):
        """Initialize database connection and session factory"""
        self.database_url = database_url
        self.engine = create_async_engine(
            database_url,
            pool_size=20,
            max_overflow=10,
            echo=False
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init_db(self):
        """Initialize database schema and extensions"""
        async with self.engine.begin() as conn:
            try:
                # Enable vector extension if available
                if HAVE_PGVECTOR:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    logger.info("Enabled pgvector extension")
                
                # Enable trigram similarity for text search
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                
                # Create tables
                await conn.run_sync(Base.metadata.create_all)
                
                # Create text search configuration
                await conn.execute(text("""
                    DO $$ 
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_ts_config 
                            WHERE cfgname = 'theorem_search'
                        ) THEN
                            CREATE TEXT SEARCH CONFIGURATION theorem_search (COPY = english);
                            ALTER TEXT SEARCH CONFIGURATION theorem_search
                                ALTER MAPPING FOR word, asciiword WITH english_stem;
                        END IF;
                    END $$;
                """))
                
                logger.info("Database initialization completed successfully")
                
            except Exception as e:
                logger.error(f"Error initializing database: {e}")
                raise

    @asynccontextmanager
    async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with proper cleanup"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            await session.close()

    async def store_theorem(
        self,
        session: AsyncSession,
        theorem_data: TheoremCreate
    ) -> str:
        """
        Store a theorem and its related entities
        
        Args:
            session: Database session
            theorem_data: Theorem creation data
            
        Returns:
            str: ID of created theorem
        """
        try:
            # Create or get concepts
            concepts = []
            for concept_id in theorem_data.concepts:
                concept = await session.get(Concept, concept_id)
                if concept:
                    concepts.append(concept)

            # Create theorem
            theorem = Theorem(
                theorem_type=theorem_data.theorem_type,
                theorem_number=theorem_data.theorem_number,
                theorem_title=theorem_data.theorem_title,
                theorem_text=theorem_data.theorem_text,
                proof_text=theorem_data.proof_text,
                page_number=theorem_data.page_number,
                source_id=theorem_data.source_id,
                proof_techniques=theorem_data.proof_techniques,
                difficulty_score=theorem_data.difficulty_score,
                spatial_info=theorem_data.spatial_info,
                formatting=theorem_data.formatting,
                ontology_mapping=theorem_data.ontology_mapping,
                theorem_embedding=theorem_data.theorem_embedding,
                proof_embedding=theorem_data.proof_embedding
            )
            
            # Add concepts
            theorem.concepts = concepts
            
            session.add(theorem)
            await session.commit()
            await session.refresh(theorem)
            
            return theorem.id

        except Exception as e:
            await session.rollback()
            logger.error(f"Error storing theorem: {e}")
            raise

    async def bulk_store_theorems(
        self,
        session: AsyncSession,
        theorems: List[TheoremCreate]
    ) -> List[str]:
        """
        Bulk store multiple theorems efficiently
        
        Args:
            session: Database session
            theorems: List of theorem creation data
            
        Returns:
            List[str]: IDs of created theorems
        """
        try:
            # Collect all unique concepts
            concept_ids = {
                concept_id
                for theorem in theorems
                for concept_id in theorem.concepts
            }
            
            # Get existing concepts
            concept_map = {}
            for concept_id in concept_ids:
                concept = await session.get(Concept, concept_id)
                if concept:
                    concept_map[concept_id] = concept

            # Create theorem objects
            theorem_objects = []
            for theorem_data in theorems:
                theorem = Theorem(
                    theorem_type=theorem_data.theorem_type,
                    theorem_number=theorem_data.theorem_number,
                    theorem_title=theorem_data.theorem_title,
                    theorem_text=theorem_data.theorem_text,
                    proof_text=theorem_data.proof_text,
                    page_number=theorem_data.page_number,
                    source_id=theorem_data.source_id,
                    proof_techniques=theorem_data.proof_techniques,
                    difficulty_score=theorem_data.difficulty_score,
                    spatial_info=theorem_data.spatial_info,
                    formatting=theorem_data.formatting,
                    ontology_mapping=theorem_data.ontology_mapping,
                    theorem_embedding=theorem_data.theorem_embedding,
                    proof_embedding=theorem_data.proof_embedding
                )
                
                # Add concepts
                theorem.concepts = [
                    concept_map[c_id]
                    for c_id in theorem_data.concepts
                    if c_id in concept_map
                ]
                
                theorem_objects.append(theorem)

            # Bulk insert
            session.add_all(theorem_objects)
            await session.commit()
            
            return [t.id for t in theorem_objects]

        except Exception as e:
            await session.rollback()
            logger.error(f"Error in bulk theorem storage: {e}")
            raise

    async def vector_search(
        self,
        session: AsyncSession,
        embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[Theorem]:
        """
        Find similar theorems using vector similarity search
        
        Args:
            session: Database session
            embedding: Vector to search for
            limit: Maximum number of results
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            List[Theorem]: Similar theorems
        """
        if not HAVE_PGVECTOR:
            logger.warning("Vector search unavailable - pgvector not installed")
            return []

        try:
            # Using pgvector's L2 distance
            result = await session.execute(f"""
                SELECT t.*, 1 - (theorem_embedding <-> :embedding) as similarity
                FROM theorems t
                WHERE 1 - (theorem_embedding <-> :embedding) > :min_similarity
                ORDER BY theorem_embedding <-> :embedding
                LIMIT :limit
            """, {
                "embedding": embedding,
                "min_similarity": min_similarity,
                "limit": limit
            })
            
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []

    async def get_theorem_graph(
        self,
        session: AsyncSession,
        theorem_id: str,
        depth: int = 2
    ) -> List[GraphEdge]:
        """
        Get the theorem dependency graph up to a certain depth
        
        Args:
            session: Database session
            theorem_id: Root theorem ID
            depth: Maximum depth to traverse
            
        Returns:
            List[GraphEdge]: Graph edges
        """
        edges = []
        visited = set()

        async def traverse(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited:
                return
            visited.add(current_id)

            result = await session.execute(
                select(GraphEdge).where(
                    (GraphEdge.from_theorem_id == current_id) |
                    (GraphEdge.to_theorem_id == current_id)
                )
            )
            
            current_edges = result.scalars().all()
            edges.extend(current_edges)

            for edge in current_edges:
                next_id = edge.to_theorem_id if edge.from_theorem_id == current_id \
                         else edge.from_theorem_id
                await traverse(next_id, current_depth + 1)

        await traverse(theorem_id, 0)
        return edges

    async def full_text_search(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 10
    ) -> List[Theorem]:
        """
        Perform full-text search on theorems and proofs
        
        Args:
            session: Database session
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List[Theorem]: Matching theorems
        """
        result = await session.execute(f"""
            SELECT t.*, 
                   ts_rank_cd(to_tsvector('english', theorem_text || ' ' || proof_text),
                             plainto_tsquery('english', :query)) as rank
            FROM theorems t
            WHERE to_tsvector('english', theorem_text || ' ' || proof_text) @@ 
                  plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """, {
            "query": query,
            "limit": limit
        })
        
        return result.scalars().all()

    async def get_similar_concepts(
        self,
        session: AsyncSession,
        concept_name: str,
        limit: int = 5
    ) -> List[Concept]:
        """
        Find similar concepts using trigram similarity
        
        Args:
            session: Database session
            concept_name: Name to search for
            limit: Maximum number of results
            
        Returns:
            List[Concept]: Similar concepts
        """
        result = await session.execute(f"""
            SELECT c.*, similarity(name, :concept_name) as sim
            FROM concepts c
            WHERE similarity(name, :concept_name) > 0.3
            ORDER BY sim DESC
            LIMIT :limit
        """, {
            "concept_name": concept_name,
            "limit": limit
        })
        
        return result.scalars().all()

    async def get_concept_hierarchy(
        self,
        session: AsyncSession,
        concept_id: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get concept hierarchy (parents and children)
        
        Args:
            session: Database session
            concept_id: Root concept ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Dict containing hierarchy information
        """
        concept = await session.get(Concept, concept_id)
        if not concept:
            return {}

        hierarchy = {
            "id": concept.id,
            "name": concept.name,
            "parents": [],
            "children": []
        }

        # Get parent concepts
        if concept.parent_concepts and max_depth > 0:
            for parent_id in concept.parent_concepts:
                parent = await self.get_concept_hierarchy(
                    session, parent_id, max_depth - 1
                )
                if parent:
                    hierarchy["parents"].append(parent)

        # Find child concepts
        result = await session.execute(
            select(Concept).where(
                func.array_position(Concept.parent_concepts, concept_id).isnot(None)
            )
        )
        children = result.scalars().all()

        if children and max_depth > 0:
            for child in children:
                child_hierarchy = await self.get_concept_hierarchy(
                    session, child.id, max_depth - 1
                )
                hierarchy["children"].append(child_hierarchy)

        return hierarchy