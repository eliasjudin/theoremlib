"""
Test suite for metadata annotation functionality
"""

import pytest
from pathlib import Path
import json
from ..annotation.metadata_annotation import (
    MetadataAnnotator, TheoremAnnotation, SourceMetadata,
    ProofTechnique, MathConcept
)
from .test_core import SAMPLE_THEOREM, SAMPLE_PROOF

@pytest.fixture
def annotator(temp_ontology):
    """Create metadata annotator with test ontology"""
    return MetadataAnnotator(temp_ontology)

@pytest.fixture
def source_metadata():
    """Create sample source metadata"""
    return SourceMetadata(
        source="Test Journal",
        author="Test Author",
        page_number=1,
        title="Test Paper",
        theorem_number="1",
        year=2024
    )

@pytest.mark.asyncio
async def test_proof_technique_identification(annotator, source_metadata):
    """Test identification of proof techniques"""
    # Test contradiction proof
    proof = "Proof. Suppose, for contradiction, that x is irrational..."
    annotation = await annotator.annotate_theorem(
        SAMPLE_THEOREM,
        proof,
        source_metadata
    )
    assert ProofTechnique.CONTRADICTION in annotation.proof_techniques

    # Test induction proof
    proof = "Proof. We proceed by induction. Base case n=1..."
    annotation = await annotator.annotate_theorem(
        SAMPLE_THEOREM,
        proof,
        source_metadata
    )
    assert ProofTechnique.INDUCTION in annotation.proof_techniques

    # Test direct proof
    annotation = await annotator.annotate_theorem(
        SAMPLE_THEOREM,
        SAMPLE_PROOF,
        source_metadata
    )
    assert ProofTechnique.DIRECT in annotation.proof_techniques

@pytest.mark.asyncio
async def test_concept_extraction(annotator, source_metadata):
    """Test extraction of mathematical concepts"""
    annotation = await annotator.annotate_theorem(
        "Theorem 1. Given a triangle in Euclidean space...",
        SAMPLE_PROOF,
        source_metadata
    )
    
    # Should identify geometry concepts
    assert any(c.category == "geometry" for c in annotation.key_concepts)
    
    # Should identify related concepts
    assert len(annotation.prerequisites) > 0
    
    # Test concept relationships
    assert annotation.ontology_mapping["primary_domain"] == "geometry"

@pytest.mark.asyncio
async def test_difficulty_calculation(annotator, source_metadata):
    """Test difficulty score calculation"""
    # Simple direct proof
    simple = await annotator.annotate_theorem(
        "Theorem 1. x = x.",
        "Proof. Obvious. □",
        source_metadata
    )
    
    # Complex proof with multiple techniques
    complex = await annotator.annotate_theorem(
        "Theorem 2. Complex statement...",
        """Proof. First, by contradiction... 
        Then, using induction... 
        Finally, probabilistic argument... □""",
        source_metadata
    )
    
    assert simple.difficulty_score < complex.difficulty_score
    assert 0 <= simple.difficulty_score <= 1
    assert 0 <= complex.difficulty_score <= 1

@pytest.mark.asyncio
async def test_confidence_scoring(annotator, source_metadata):
    """Test confidence score calculation"""
    # High confidence case - clear theorem and proof
    high_conf = await annotator.annotate_theorem(
        SAMPLE_THEOREM,
        SAMPLE_PROOF,
        source_metadata
    )
    
    # Low confidence case - ambiguous content
    low_conf = await annotator.annotate_theorem(
        "Statement (maybe a theorem?)...",
        "Here's some related text...",
        source_metadata
    )
    
    assert high_conf.confidence_score > low_conf.confidence_score
    assert 0 <= low_conf.confidence_score <= 1
    assert 0 <= high_conf.confidence_score <= 1

@pytest.mark.asyncio
async def test_prerequisites_identification(annotator, source_metadata):
    """Test identification of prerequisite concepts"""
    annotation = await annotator.annotate_theorem(
        "Theorem 1. Using concepts from linear algebra...",
        "Proof. Applying matrix operations... □",
        source_metadata
    )
    
    prereqs = annotation.prerequisites
    assert len(prereqs) > 0
    # Should include basic algebra concepts
    assert any("algebra" in p.lower() for p in prereqs)

@pytest.mark.asyncio
async def test_semantic_similarity(annotator, source_metadata):
    """Test semantic similarity matching with transformers"""
    if annotator.sentence_model is not None:
        annotation = await annotator.annotate_theorem(
            "Theorem 1. A statement about geometric series...",
            "Proof using convergence tests... □",
            source_metadata
        )
        
        # Should find related concepts even without exact matches
        assert len(annotation.key_concepts) > 0
        assert annotation.confidence_score > 0.5

@pytest.mark.asyncio
async def test_error_handling(annotator, source_metadata):
    """Test handling of invalid inputs"""
    # Empty theorem
    with pytest.raises(ValueError):
        await annotator.annotate_theorem(
            "",
            SAMPLE_PROOF,
            source_metadata
        )
    
    # Invalid source metadata
    with pytest.raises(ValueError):
        invalid_metadata = source_metadata.copy()
        invalid_metadata.page_number = -1  # Invalid page number
        await annotator.annotate_theorem(
            SAMPLE_THEOREM,
            SAMPLE_PROOF,
            invalid_metadata
        )