"""
Core test module for shared fixtures and utilities.
"""

import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from ..database.models import Source, Theorem, ProofTechniqueEnum
from ..database.database_loader import DatabaseLoader
from ..annotation.metadata_annotation import MetadataAnnotator
from ..extraction.content_extraction import ContentExtractor
from ..ingest.ingest import PDFIngestor
from ..main_pipeline import PipelineCoordinator

# Test data constants
SAMPLE_THEOREM = """
Theorem 1 (Pythagorean Theorem). In a right triangle with legs a and b and 
hypotenuse c, we have a² + b² = c².
"""

SAMPLE_PROOF = """
Proof. Consider the square of side length a + b containing four copies of 
the right triangle arranged to form an inner square of side length c. 
The area can be calculated in two ways:
(a + b)² = c² + 4(ab/2)
a² + 2ab + b² = c² + 2ab
Therefore, a² + b² = c². □
"""

@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Provide sample document metadata"""
    return {
        "source": "Sample Mathematics Journal",
        "title": "Introduction to Triangle Theorems",
        "author": "Test Author",
        "page_count": 10,
        "theorem_number": "1",
        "upload_time": datetime.utcnow()
    }

@pytest.fixture
def temp_ontology(tmp_path) -> Path:
    """Create a temporary ontology file for testing"""
    ontology_data = {
        "geometry": {
            "name": "Geometry",
            "category": "geometry",
            "parent_concepts": [],
            "related_concepts": ["trigonometry"]
        },
        "trigonometry": {
            "name": "Trigonometry",
            "category": "geometry",
            "parent_concepts": ["geometry"],
            "related_concepts": ["geometry"]
        }
    }
    ontology_file = tmp_path / "test_ontology.json"
    with open(ontology_file, "w") as f:
        json.dump(ontology_data, f)
    return ontology_file

@pytest.fixture
async def db_session():
    """Get test database session"""
    db_url = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test_theoremlib")
    loader = DatabaseLoader(db_url)
    async with loader.get_db() as session:
        await loader.init_db()
        yield session

@pytest.fixture
def sample_theorem_proof():
    """Provide sample theorem-proof pair"""
    return SAMPLE_THEOREM, SAMPLE_PROOF

# Helper function for testing vector operations
def vector_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 * norm2 else 0.0

class TestHelper:
    """Helper class for test utilities"""
    @staticmethod
    def verify_theorem_metadata(theorem: Theorem) -> bool:
        """Verify that a theorem has required metadata"""
        return all([
            theorem.theorem_type is not None,
            theorem.theorem_text is not None,
            theorem.proof_text is not None,
            isinstance(theorem.proof_techniques, list),
            isinstance(theorem.difficulty_score, float),
            0 <= theorem.difficulty_score <= 1
        ])

    @staticmethod
    def verify_graph_edge(
        source_id: str,
        target_id: str,
        edge_type: str,
        graph_edges: list
    ) -> bool:
        """Verify that a specific edge exists in graph edges"""
        return any(
            e.from_theorem_id == source_id and
            e.to_theorem_id == target_id and
            e.edge_type == edge_type
            for e in graph_edges
        )