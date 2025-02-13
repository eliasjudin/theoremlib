"""
Database Loader Module

Handles database operations, connection management, and migrations using SQLAlchemy and Alembic.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from contextlib import contextmanager
from typing import Generator, List
import logging
from pathlib import Path

from .models import Base, Theorem, Concept, Source, GraphEdge

logger = logging.getLogger(__name__)

class DatabaseLoader:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_async_engine(database_url)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init_db(self):
        """Initialize database schema"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_db(self) -> AsyncSession:
        """Get database session"""
        async with self.SessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()

    async def store_theorem(self, session: AsyncSession, theorem: Theorem) -> str:
        """Store a theorem and its related entities"""
        try:
            # Store or update related concepts
            for concept in theorem.concepts:
                existing = await session.get(Concept, concept.id)
                if not existing:
                    session.add(concept)

            # Store the theorem
            session.add(theorem)
            await session.commit()
            return theorem.id
        except Exception as e:
            await session.rollback()
            logger.error(f"Error storing theorem: {e}")
            raise

    async def bulk_store_theorems(
        self, 
        session: AsyncSession, 
        theorems: List[Theorem]
    ) -> List[str]:
        """Bulk store multiple theorems efficiently"""
        try:
            # Collect all unique concepts
            all_concepts = {
                concept.id: concept
                for theorem in theorems
                for concept in theorem.concepts
            }

            # Bulk insert concepts
            for concept in all_concepts.values():
                existing = await session.get(Concept, concept.id)
                if not existing:
                    session.add(concept)

            # Bulk insert theorems
            session.add_all(theorems)
            await session.commit()
            
            return [theorem.id for theorem in theorems]
        except Exception as e:
            await session.rollback()
            logger.error(f"Error in bulk theorem storage: {e}")
            raise

    async def vector_search(
        self,
        session: AsyncSession,
        embedding: List[float],
        limit: int = 10
    ) -> List[Theorem]:
        """Find similar theorems using vector similarity search"""
        # Using pgvector's L2 distance
        result = await session.execute(f"""
            SELECT t.* FROM theorems t
            ORDER BY theorem_embedding <-> :embedding
            LIMIT :limit
        """, {
            "embedding": embedding,
            "limit": limit
        })
        return result.scalars().all()

    async def get_theorem_graph(
        self,
        session: AsyncSession,
        theorem_id: str,
        depth: int = 2
    ) -> List[GraphEdge]:
        """Get the theorem dependency graph up to a certain depth"""
        edges = []
        visited = set()

        async def traverse(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited:
                return
            visited.add(current_id)

            result = await session.execute(f"""
                SELECT * FROM graph_edges
                WHERE from_theorem_id = :theorem_id
                   OR to_theorem_id = :theorem_id
            """, {"theorem_id": current_id})
            
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
        """Perform full-text search on theorems and proofs"""
        result = await session.execute(f"""
            SELECT t.* FROM theorems t
            WHERE to_tsvector('english', theorem_text || ' ' || proof_text) @@ 
                  plainto_tsquery('english', :query)
            LIMIT :limit
        """, {
            "query": query,
            "limit": limit
        })
        return result.scalars().all()