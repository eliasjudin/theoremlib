"""
Test suite for content extraction functionality
"""

import pytest
from ..extraction.content_extraction import ContentExtractor, TheoremType, TheoremProofPair
from .test_core import SAMPLE_THEOREM, SAMPLE_PROOF

@pytest.fixture
def extractor():
    """Create content extractor instance"""
    return ContentExtractor()

def test_theorem_identification(extractor):
    """Test identification of theorem statements"""
    text = SAMPLE_THEOREM + "\nSome other text\n" + "Lemma 2. Another statement."
    
    pairs = extractor.extract_pairs(
        text,
        spatial_info={},
        page_num=1
    )
    
    assert len(pairs) == 2
    assert pairs[0].theorem_type == TheoremType.THEOREM
    assert pairs[0].theorem_number == "1"
    assert "Pythagorean" in pairs[0].theorem_title
    assert pairs[1].theorem_type == TheoremType.LEMMA
    assert pairs[1].theorem_number == "2"

def test_proof_matching(extractor):
    """Test matching proofs with theorems"""
    text = SAMPLE_THEOREM + "\n" + SAMPLE_PROOF
    
    pairs = extractor.extract_pairs(
        text,
        spatial_info={},
        page_num=1
    )
    
    assert len(pairs) == 1
    assert "□" in pairs[0].proof_text
    assert pairs[0].confidence_score > 0.8

def test_spatial_info_extraction(extractor):
    """Test extraction of spatial layout information"""
    spatial_info = {
        "blocks": [
            {
                "span_start": 0,
                "span_end": 100,
                "font": "CMR10",
                "size": 12,
                "style": "bold"
            }
        ]
    }
    
    pairs = extractor.extract_pairs(
        SAMPLE_THEOREM + "\n" + SAMPLE_PROOF,
        spatial_info=spatial_info,
        page_num=1
    )
    
    assert len(pairs) == 1
    assert "font" in pairs[0].formatting
    assert pairs[0].formatting["primary_font"] == "CMR10"

def test_multipage_extraction(extractor):
    """Test extraction across page boundaries"""
    text1 = SAMPLE_THEOREM + "\nProof. Consider the square"
    text2 = "of side length a + b containing... □"
    
    # Extract from first page
    pairs1 = extractor.extract_pairs(text1, {}, 1)
    assert len(pairs1) == 1
    assert not pairs1[0].proof_text.endswith("□")
    
    # Extract from second page
    pairs2 = extractor.extract_pairs(text2, {}, 2)
    assert len(pairs2) == 0  # Should not extract incomplete proof
    
    # Extract with combined text
    pairs = extractor.extract_pairs(text1 + text2, {}, 1)
    assert len(pairs) == 1
    assert pairs[0].proof_text.endswith("□")

def test_complex_formatting(extractor):
    """Test extraction with complex formatting"""
    spatial_info = {
        "blocks": [
            {
                "span_start": 0,
                "span_end": 50,
                "font": "CMR10",
                "size": 12,
                "style": "bold",
                "alignment": "center"
            },
            {
                "span_start": 51,
                "span_end": 200,
                "font": "CMR10",
                "size": 10,
                "style": "italic",
                "alignment": "left"
            }
        ]
    }
    
    pairs = extractor.extract_pairs(
        SAMPLE_THEOREM + "\n" + SAMPLE_PROOF,
        spatial_info=spatial_info,
        page_num=1
    )
    
    assert len(pairs) == 1
    assert "alignment" in pairs[0].spatial_info
    assert pairs[0].formatting["styles"].get("bold") is not None
    assert pairs[0].formatting["styles"].get("italic") is not None

def test_error_handling(extractor):
    """Test handling of malformed input"""
    # Test with empty text
    pairs = extractor.extract_pairs("", {}, 1)
    assert len(pairs) == 0
    
    # Test with malformed theorem
    pairs = extractor.extract_pairs("Theorem without number or period", {}, 1)
    assert len(pairs) == 0
    
    # Test with invalid spatial info
    with pytest.raises(Exception):
        extractor.extract_pairs(
            SAMPLE_THEOREM,
            {"blocks": [{"invalid": "data"}]},
            1
        )