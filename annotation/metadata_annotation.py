"""
Metadata Annotation Module

Enriches theorem-proof pairs with comprehensive metadata using the Cambridge
Math Ontology and performs proof technique classification.
"""

from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import re
import json
import logging
from pathlib import Path
import numpy as np
from importlib import util
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers, use simple fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    HAVE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence-transformers not available, falling back to basic text matching")
    HAVE_TRANSFORMERS = False

class ProofTechnique(str, Enum):
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CONSTRUCTIVE = "constructive"
    PROBABILISTIC = "probabilistic"
    ALGEBRAIC = "algebraic"
    GEOMETRIC = "geometric"

class MathConcept(BaseModel):
    """Representation of a mathematical concept from the ontology"""
    id: str = Field(..., description="Unique identifier for the concept")
    name: str = Field(..., description="Human-readable name of the concept")
    category: str = Field(..., description="Mathematical category (e.g., algebra, analysis)")
    parent_concepts: List[str] = Field(default_factory=list, description="List of parent concept IDs")
    related_concepts: List[str] = Field(default_factory=list, description="List of related concept IDs")

    class Config:
        frozen = True

class SourceMetadata(BaseModel):
    """Source document metadata"""
    source: str = Field(..., description="Source of the theorem (e.g., paper, book)")
    author: str = Field(..., description="Author(s) of the source")
    page_number: int = Field(..., description="Page number where theorem appears")
    title: str = Field(..., description="Title of the source document")
    theorem_number: Optional[str] = Field(None, description="Original theorem number in source")
    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Publication venue")

    class Config:
        frozen = True

class TheoremAnnotation(BaseModel):
    """Complete annotation for a theorem-proof pair"""
    source_metadata: SourceMetadata = Field(..., description="Source document metadata")
    proof_techniques: List[ProofTechnique] = Field(..., description="List of proof techniques used")
    key_concepts: List[MathConcept] = Field(..., description="Key mathematical concepts from ontology")
    difficulty_score: float = Field(..., ge=0.0, le=1.0, description="Computed difficulty score")
    proof_length: int = Field(..., gt=0, description="Length of proof in words")
    prerequisites: List[str] = Field(..., description="List of prerequisite concept IDs")
    ontology_mapping: Dict[str, Any] = Field(..., description="Mapping to Cambridge Math Ontology")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in annotations")

    class Config:
        frozen = True

class MetadataAnnotator:
    def __init__(self, ontology_path: Path):
        """Initialize with Cambridge Math Ontology and models"""
        self.ontology = self._load_ontology(ontology_path)
        
        # Initialize sentence transformer if available
        self.sentence_model = None
        if HAVE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer model: {e}")
                
        # Load technique classification indicators
        self.technique_indicators = {
            ProofTechnique.DIRECT: [
                "clearly", "observe that", "note that", "directly",
                "it follows that", "therefore", "hence"
            ],
            ProofTechnique.CONTRADICTION: [
                "contradiction", "suppose not", "assume contrary",
                "leads to a contradiction", "cannot be true"
            ],
            ProofTechnique.INDUCTION: [
                "base case", "inductive step", "inductive hypothesis",
                "assume true for k", "prove for k+1"
            ],
            ProofTechnique.CONSTRUCTIVE: [
                "construct", "define", "let us build",
                "we can create", "existence"
            ],
            ProofTechnique.PROBABILISTIC: [
                "probability", "random", "expected value",
                "with high probability", "on average"
            ],
            ProofTechnique.ALGEBRAIC: [
                "algebra", "polynomial", "equation",
                "matrix", "linear", "factor"
            ],
            ProofTechnique.GEOMETRIC: [
                "geometric", "triangle", "circle",
                "angle", "distance", "space"
            ]
        }

    def _load_ontology(self, path: Path) -> Dict[str, MathConcept]:
        """Load and parse the Cambridge Math Ontology"""
        try:
            with open(path) as f:
                data = json.load(f)
            return {
                concept_id: MathConcept(
                    id=concept_id,
                    **{k: v for k, v in concept_data.items() if k != 'id'}
                )
                for concept_id, concept_data in data.items()
                if not concept_id.startswith('proof_techniques')  # Skip proof technique entries
            }
        except Exception as e:
            logger.error(f"Error loading ontology from {path}: {e}")
            raise ValueError(f"Failed to load ontology: {e}")

    async def annotate_theorem(
        self,
        theorem_text: str,
        proof_text: str,
        source_metadata: SourceMetadata
    ) -> TheoremAnnotation:
        """
        Generate comprehensive annotations for a theorem-proof pair

        Args:
            theorem_text: The theorem statement text
            proof_text: The proof text
            source_metadata: Metadata about the source document

        Returns:
            TheoremAnnotation object with complete metadata
        """
        try:
            # Generate embeddings if possible
            theorem_embedding = None
            if self.sentence_model is not None:
                try:
                    theorem_embedding = self.sentence_model.encode(theorem_text)
                except Exception as e:
                    logger.warning(f"Failed to generate theorem embedding: {e}")
            
            # Identify proof techniques
            techniques = self._identify_proof_techniques(proof_text)
            
            # Extract key mathematical concepts
            concepts = self._extract_key_concepts(theorem_text, proof_text, theorem_embedding)
            
            # Calculate metrics
            difficulty = self._calculate_difficulty(proof_text, techniques, concepts)
            prereqs = self._identify_prerequisites(concepts, techniques)
            
            # Map to ontology
            ontology_mapping = self._map_to_ontology(
                theorem_text,
                proof_text,
                concepts,
                techniques
            )

            return TheoremAnnotation(
                source_metadata=source_metadata,
                proof_techniques=techniques,
                key_concepts=concepts,
                difficulty_score=difficulty,
                proof_length=len(proof_text.split()),
                prerequisites=prereqs,
                ontology_mapping=ontology_mapping,
                confidence_score=self._calculate_confidence(techniques, concepts)
            )

        except Exception as e:
            logger.error(f"Error annotating theorem: {e}")
            logger.error(f"Theorem text: {theorem_text[:100]}...")
            raise

    def _identify_proof_techniques(self, proof_text: str) -> List[ProofTechnique]:
        """Identify the proof techniques used in the proof"""
        techniques = []
        lower_proof = proof_text.lower()
        
        # Use indicator phrases to identify techniques
        for technique, indicators in self.technique_indicators.items():
            if any(ind.lower() in lower_proof for ind in indicators):
                techniques.append(technique)
        
        # If no specific technique identified, assume direct proof
        if not techniques:
            techniques = [ProofTechnique.DIRECT]
            
        return techniques

    def _extract_key_concepts(
        self,
        theorem_text: str,
        proof_text: str,
        theorem_embedding: Optional[np.ndarray] = None
    ) -> List[MathConcept]:
        """Extract key mathematical concepts using ontology and semantic similarity"""
        concepts = set()
        combined_text = f"{theorem_text} {proof_text}".lower()
        
        # Direct keyword matching
        for concept in self.ontology.values():
            if concept.name.lower() in combined_text:
                concepts.add(concept)
                
            # Check related terms
            for related in concept.related_concepts:
                if related.lower() in combined_text:
                    concepts.add(concept)
        
        # Semantic similarity matching if transformer model is available
        if len(concepts) < 3 and self.sentence_model is not None and theorem_embedding is not None:
            try:
                concept_embeddings = {
                    concept: self.sentence_model.encode(concept.name)
                    for concept in self.ontology.values()
                }
                
                similarities = {
                    concept: np.dot(theorem_embedding, emb) / (np.linalg.norm(theorem_embedding) * np.linalg.norm(emb))
                    for concept, emb in concept_embeddings.items()
                }
                
                # Add highly similar concepts
                for concept, similarity in similarities.items():
                    if similarity > 0.7 and concept not in concepts:
                        concepts.add(concept)
            except Exception as e:
                logger.warning(f"Error in semantic similarity matching: {e}")
                # Fall back to simple matching only
        
        return list(concepts)

    def _calculate_difficulty(
        self,
        proof_text: str,
        techniques: List[ProofTechnique],
        concepts: List[MathConcept]
    ) -> float:
        """Calculate a difficulty score based on multiple factors"""
        score = 0.0
        
        # Base difficulty from proof length
        words = len(proof_text.split())
        score += min(words / 1000, 0.3)  # Cap at 0.3
        
        # Technique complexity
        technique_scores = {
            ProofTechnique.DIRECT: 0.1,
            ProofTechnique.CONSTRUCTIVE: 0.2,
            ProofTechnique.INDUCTION: 0.3,
            ProofTechnique.CONTRADICTION: 0.3,
            ProofTechnique.PROBABILISTIC: 0.4,
            ProofTechnique.ALGEBRAIC: 0.25,
            ProofTechnique.GEOMETRIC: 0.25
        }
        
        if techniques:
            score += sum(technique_scores[t] for t in techniques) / len(techniques)
        
        # Concept complexity
        if concepts:
            concept_depth = max(len(c.parent_concepts) for c in concepts)
            score += min(concept_depth * 0.1, 0.3)  # Cap at 0.3
        
        return min(score, 1.0)  # Normalize to [0,1]

    def _identify_prerequisites(
        self,
        concepts: List[MathConcept],
        techniques: List[ProofTechnique]
    ) -> List[str]:
        """Identify prerequisite concepts and theorems"""
        prereqs = set()
        
        # Add parent concepts as prerequisites
        for concept in concepts:
            prereqs.update(concept.parent_concepts)
        
        # Add technique-specific prerequisites
        technique_prereqs = {
            ProofTechnique.INDUCTION: ["mathematical_induction", "natural_numbers"],
            ProofTechnique.CONTRADICTION: ["logic", "negation"],
            ProofTechnique.PROBABILISTIC: ["probability", "expectation"],
            ProofTechnique.ALGEBRAIC: ["algebra", "equations"],
            ProofTechnique.GEOMETRIC: ["geometry", "euclidean_space"]
        }
        
        for technique in techniques:
            if technique in technique_prereqs:
                prereqs.update(technique_prereqs[technique])
        
        return list(prereqs)

    def _map_to_ontology(
        self,
        theorem_text: str,
        proof_text: str,
        concepts: List[MathConcept],
        techniques: List[ProofTechnique]
    ) -> Dict[str, Any]:
        """Map theorem content to the Cambridge Math Ontology structure"""
        return {
            "primary_domain": self._determine_primary_domain(concepts),
            "related_domains": [c.category for c in concepts],
            "proof_methods": [t.value for t in techniques],
            "logical_structure": {
                "has_conditions": bool(re.search(r"if|suppose|assume|let|given", theorem_text, re.I)),
                "has_construction": bool(re.search(r"construct|define|consider|take", proof_text, re.I)),
                "has_cases": bool(re.search(r"case [0-9]|first case|second case", proof_text, re.I))
            },
            "concept_relations": [
                {
                    "from": c1.id,
                    "to": c2.id,
                    "type": "related"
                }
                for i, c1 in enumerate(concepts)
                for c2 in concepts[i+1:]
                if c2.id in c1.related_concepts
            ]
        }

    def _determine_primary_domain(self, concepts: List[MathConcept]) -> str:
        """Determine the primary mathematical domain of the theorem"""
        if not concepts:
            return "general"
            
        # Count concept categories
        category_counts = {}
        for concept in concepts:
            category_counts[concept.category] = category_counts.get(concept.category, 0) + 1
            
        # Return most common category
        return max(category_counts.items(), key=lambda x: x[1])[0]

    def _calculate_confidence(
        self,
        techniques: List[ProofTechnique],
        concepts: List[MathConcept]
    ) -> float:
        """Calculate confidence score for the annotations"""
        confidence = 0.0
        
        # Technique identification confidence
        if techniques:
            confidence += 0.4  # Base confidence for finding techniques
            if len(techniques) > 1:
                confidence += 0.1  # Bonus for multiple techniques
                
        # Concept identification confidence
        if concepts:
            concept_score = min(len(concepts) * 0.1, 0.4)  # Up to 0.4 for concepts
            confidence += concept_score
            
            # Check concept relationships
            has_relationships = any(
                c2.id in c1.related_concepts
                for i, c1 in enumerate(concepts)
                for c2 in concepts[i+1:]
            )
            if has_relationships:
                confidence += 0.1  # Bonus for finding related concepts
                
        return min(confidence, 1.0)  # Normalize to [0,1]