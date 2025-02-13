"""
Content Extraction Module

Handles the identification and extraction of theorem-proof pairs from processed text,
preserving formatting and spatial layout information.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel
import re
from dataclasses import dataclass
from enum import Enum

class TheoremType(str, Enum):
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"

@dataclass
class TextSpan:
    start: int
    end: int
    text: str
    formatting: Dict

class TheoremProofPair(BaseModel):
    """Model representing a theorem-proof pair with metadata"""
    theorem_type: TheoremType
    theorem_number: str
    theorem_title: Optional[str]
    theorem_text: str
    proof_text: str
    page_number: int
    spatial_info: Dict
    formatting: Dict
    confidence_score: float

class ContentExtractor:
    def __init__(self):
        # Regular expressions for identifying different theorem types
        self.theorem_patterns = {
            TheoremType.THEOREM: r"Theorem\s+(\d+\.?\d*)",
            TheoremType.LEMMA: r"Lemma\s+(\d+\.?\d*)",
            TheoremType.PROPOSITION: r"Proposition\s+(\d+\.?\d*)",
            TheoremType.COROLLARY: r"Corollary\s+(\d+\.?\d*)"
        }
        self.proof_pattern = r"Proof\.?\s+"

    def extract_pairs(self, text: str, spatial_info: Dict, page_num: int) -> List[TheoremProofPair]:
        """Extract theorem-proof pairs from processed text"""
        pairs = []
        current_position = 0
        
        while current_position < len(text):
            # Find next theorem
            theorem_match = None
            theorem_type = None
            
            for t_type, pattern in self.theorem_patterns.items():
                match = re.search(pattern, text[current_position:])
                if match and (theorem_match is None or match.start() < theorem_match.start()):
                    theorem_match = match
                    theorem_type = t_type

            if not theorem_match:
                break

            # Find corresponding proof
            proof_start = re.search(self.proof_pattern, text[current_position + theorem_match.end():])
            if not proof_start:
                current_position += theorem_match.end()
                continue

            # Extract theorem text
            theorem_start = current_position + theorem_match.start()
            theorem_end = current_position + theorem_match.end() + proof_start.start()
            theorem_text = text[theorem_start:theorem_end].strip()

            # Look for the end of the proof (next theorem or end of text)
            next_theorem_match = None
            for pattern in self.theorem_patterns.values():
                match = re.search(pattern, text[theorem_end + proof_start.end():])
                if match and (next_theorem_match is None or match.start() < next_theorem_match.start()):
                    next_theorem_match = match

            proof_end = theorem_end + proof_start.end() + \
                       (next_theorem_match.start() if next_theorem_match else len(text))
            proof_text = text[theorem_end + proof_start.end():proof_end].strip()

            # Create theorem-proof pair
            pair = TheoremProofPair(
                theorem_type=theorem_type,
                theorem_number=theorem_match.group(1),
                theorem_title=self._extract_title(theorem_text),
                theorem_text=theorem_text,
                proof_text=proof_text,
                page_number=page_num,
                spatial_info=self._extract_spatial_info(spatial_info, theorem_start, proof_end),
                formatting=self._extract_formatting(spatial_info, theorem_start, proof_end),
                confidence_score=self._calculate_confidence(theorem_text, proof_text)
            )
            pairs.append(pair)

            current_position = proof_end

        return pairs

    def _extract_title(self, theorem_text: str) -> Optional[str]:
        """Extract theorem title if present"""
        title_match = re.search(r"\((.*?)\)", theorem_text)
        return title_match.group(1) if title_match else None

    def _extract_spatial_info(self, spatial_info: Dict, start: int, end: int) -> Dict:
        """Extract relevant spatial information for the theorem-proof pair"""
        return {
            k: v for k, v in spatial_info.items()
            if start <= v.get("start", 0) <= end
        }

    def _extract_formatting(self, spatial_info: Dict, start: int, end: int) -> Dict:
        """Extract formatting information for the theorem-proof pair"""
        return {
            "styles": self._collect_styles(spatial_info, start, end),
            "layout": self._collect_layout(spatial_info, start, end)
        }

    def _calculate_confidence(self, theorem_text: str, proof_text: str) -> float:
        """Calculate confidence score for the extraction"""
        # Basic heuristic: check for expected structure and completeness
        has_theorem = any(re.search(pattern, theorem_text) 
                         for pattern in self.theorem_patterns.values())
        has_proof = bool(re.search(self.proof_pattern, proof_text))
        has_qed = proof_text.strip().endswith("□") or "QED" in proof_text.upper()
        
        confidence = 0.0
        if has_theorem:
            confidence += 0.4
        if has_proof:
            confidence += 0.4
        if has_qed:
            confidence += 0.2
            
        return confidence

    @staticmethod
    def _collect_styles(spatial_info: Dict, start: int, end: int) -> Dict:
        """Collect style information from the spatial info"""
        return {
            "fonts": {},
            "sizes": {},
            "formatting": {}
        }

    @staticmethod
    def _collect_layout(spatial_info: Dict, start: int, end: int) -> Dict:
        """Collect layout information from the spatial info"""
        return {
            "columns": [],
            "margins": {},
            "spacing": {}
        }