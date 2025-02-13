"""
Content Extraction Module

Handles the identification and extraction of theorem-proof pairs from processed text,
preserving formatting and spatial layout information.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import re
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TheoremType(str, Enum):
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    DEFINITION = "definition"

@dataclass
class TextSpan:
    start: int
    end: int
    text: str
    formatting: Dict[str, Any]

class TheoremProofPair(BaseModel):
    """Model representing a theorem-proof pair with metadata"""
    theorem_type: TheoremType
    theorem_number: Optional[str]
    theorem_title: Optional[str]
    theorem_text: str
    proof_text: str
    page_number: int
    spatial_info: Dict[str, Any]
    formatting: Dict[str, Any]
    confidence_score: float

class ContentExtractor:
    def __init__(self):
        # Enhanced patterns for theorem identification
        self.theorem_patterns = {
            TheoremType.THEOREM: [
                r"Theorem\s+(?P<number>\d+\.?\d*)\s*(?:\((?P<title>[^)]+)\))?\s*[\.:]?\s*(?P<statement>.*?)(?=\n|$)",
                r"Theorem\s*\((?P<title>[^)]+)\)\s*(?P<number>\d+\.?\d*)?\s*[\.:]?\s*(?P<statement>.*?)(?=\n|$)"
            ],
            TheoremType.LEMMA: [
                r"Lemma\s+(?P<number>\d+\.?\d*)\s*(?:\((?P<title>[^)]+)\))?\s*[\.:]?\s*(?P<statement>.*?)(?=\n|$)"
            ],
            TheoremType.PROPOSITION: [
                r"Proposition\s+(?P<number>\d+\.?\d*)\s*(?:\((?P<title>[^)]+)\))?\s*[\.:]?\s*(?P<statement>.*?)(?=\n|$)"
            ],
            TheoremType.COROLLARY: [
                r"Corollary\s+(?P<number>\d+\.?\d*)\s*(?:\((?P<title>[^)]+)\))?\s*[\.:]?\s*(?P<statement>.*?)(?=\n|$)"
            ],
            TheoremType.DEFINITION: [
                r"Definition\s+(?P<number>\d+\.?\d*)\s*(?:\((?P<title>[^)]+)\))?\s*[\.:]?\s*(?P<statement>.*?)(?=\n|$)"
            ]
        }
        self.proof_patterns = [
            r"Proof[\.:]\s*(?P<proof>.*?)(?=(?:\n\s*(?:Theorem|Lemma|Proposition|Corollary|Definition)\s)|$)",
            r"Proof\s*(?:\((?P<method>[^)]+)\))[\.:]\s*(?P<proof>.*?)(?=(?:\n\s*(?:Theorem|Lemma|Proposition|Corollary|Definition)\s)|$)"
        ]
        self.qed_markers = [r"□", r"QED", r"q\.e\.d\.", r"\[End of [Pp]roof\]"]

    def extract_pairs(self, text: str, spatial_info: Dict[str, Any], page_num: int) -> List[TheoremProofPair]:
        """
        Extract theorem-proof pairs from processed text with enhanced pattern matching
        and error handling.

        Args:
            text: The full text content to process
            spatial_info: Dictionary containing spatial layout information
            page_num: The page number being processed

        Returns:
            List of TheoremProofPair objects
        """
        pairs = []
        current_position = 0
        
        while current_position < len(text):
            # Find next theorem-like statement
            theorem_match = None
            theorem_type = None
            pattern_used = None
            
            for t_type, patterns in self.theorem_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, text[current_position:], re.DOTALL | re.MULTILINE)
                    if match and (theorem_match is None or match.start() < theorem_match.start()):
                        theorem_match = match
                        theorem_type = t_type
                        pattern_used = pattern

            if not theorem_match:
                break

            # Extract theorem components
            theorem_start = current_position + theorem_match.start()
            theorem_text = theorem_match.group()
            
            # Look for proof
            proof_text = ""
            proof_confidence = 0.0
            
            for proof_pattern in self.proof_patterns:
                proof_match = re.search(proof_pattern, text[theorem_start + len(theorem_text):], re.DOTALL)
                if proof_match:
                    proof_text = proof_match.group('proof').strip()
                    # Check for QED marker
                    for qed in self.qed_markers:
                        if re.search(qed, proof_text, re.IGNORECASE):
                            proof_confidence = 1.0
                            break
                    if proof_confidence < 1.0:
                        proof_confidence = 0.8  # Found proof but no QED marker
                    break

            if not proof_text:
                # Try to find the next theorem to bound this one's content
                next_theorem = None
                for t_patterns in self.theorem_patterns.values():
                    for pattern in t_patterns:
                        match = re.search(pattern, text[theorem_start + len(theorem_text):], re.DOTALL)
                        if match and (next_theorem is None or match.start() < next_theorem.start()):
                            next_theorem = match

                # If no next theorem, take everything until the end
                if next_theorem:
                    proof_text = text[theorem_start + len(theorem_text):theorem_start + len(theorem_text) + next_theorem.start()].strip()
                else:
                    proof_text = text[theorem_start + len(theorem_text):].strip()

            # Create TheoremProofPair
            try:
                pair = TheoremProofPair(
                    theorem_type=theorem_type,
                    theorem_number=theorem_match.group('number') if 'number' in theorem_match.groupdict() else None,
                    theorem_title=theorem_match.group('title') if 'title' in theorem_match.groupdict() else None,
                    theorem_text=theorem_text.strip(),
                    proof_text=proof_text,
                    page_number=page_num,
                    spatial_info=self._extract_spatial_info(spatial_info, theorem_start, theorem_start + len(theorem_text) + len(proof_text)),
                    formatting=self._extract_formatting(spatial_info, theorem_start, theorem_start + len(theorem_text) + len(proof_text)),
                    confidence_score=self._calculate_confidence(theorem_text, proof_text, proof_confidence)
                )
                pairs.append(pair)
            except Exception as e:
                logger.error(f"Error creating theorem-proof pair: {e}")
                logger.debug(f"Theorem text: {theorem_text[:100]}...")
                logger.debug(f"Proof text: {proof_text[:100]}...")

            # Move past this theorem-proof pair
            current_position = theorem_start + len(theorem_text) + len(proof_text)

        return pairs

    def _extract_spatial_info(self, spatial_info: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
        """Extract relevant spatial information for the theorem-proof pair"""
        relevant_info = {}
        
        # Extract blocks that overlap with our text span
        if "blocks" in spatial_info:
            relevant_blocks = []
            for block in spatial_info["blocks"]:
                block_start = block.get("span_start", 0)
                block_end = block.get("span_end", float("inf"))
                
                if (block_start <= end and block_end >= start):
                    relevant_blocks.append(block)
            
            relevant_info["blocks"] = relevant_blocks

        # Include page layout information if available
        if "layout" in spatial_info:
            relevant_info["layout"] = spatial_info["layout"]

        return relevant_info

    def _extract_formatting(self, spatial_info: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
        """Extract formatting information for the theorem-proof pair"""
        formatting = {
            "fonts": {},
            "sizes": {},
            "styles": {},
            "alignment": "left",  # default
            "indentation": 0      # default
        }

        # Extract formatting from blocks
        if "blocks" in spatial_info:
            for block in spatial_info["blocks"]:
                if block.get("span_start", 0) <= end and block.get("span_end", float("inf")) >= start:
                    if "font" in block:
                        formatting["fonts"][block["font"]] = formatting["fonts"].get(block["font"], 0) + 1
                    if "size" in block:
                        formatting["sizes"][block["size"]] = formatting["sizes"].get(block["size"], 0) + 1
                    if "style" in block:
                        formatting["styles"][block["style"]] = formatting["styles"].get(block["style"], 0) + 1

        # Determine most common formatting attributes
        for attr in ["fonts", "sizes", "styles"]:
            if formatting[attr]:
                formatting[f"primary_{attr[:-1]}"] = max(formatting[attr].items(), key=lambda x: x[1])[0]

        return formatting

    def _calculate_confidence(self, theorem_text: str, proof_text: str, proof_confidence: float) -> float:
        """Calculate confidence score for the extraction"""
        confidence = 0.0
        
        # Check theorem structure
        if any(re.search(pattern, theorem_text, re.IGNORECASE) for patterns in self.theorem_patterns.values() for pattern in patterns):
            confidence += 0.4
        
        # Check proof structure
        if proof_text and any(re.search(pattern, proof_text, re.IGNORECASE) for pattern in self.proof_patterns):
            confidence += 0.3
        
        # Factor in proof confidence
        confidence += proof_confidence * 0.3
        
        # Normalize confidence score
        return min(confidence, 1.0)