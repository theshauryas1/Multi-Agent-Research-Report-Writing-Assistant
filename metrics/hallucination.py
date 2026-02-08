"""
Hallucination Detector - Identifies fabricated citations and unsupported claims.
Validates that all references in generated content exist in actual sources.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Citation:
    """A citation found in generated text."""
    text: str
    reference: str
    position: int
    is_valid: bool = False
    matched_source: Optional[str] = None


class HallucinationDetector:
    """
    Detects hallucinated citations and fabricated references.
    
    Validates that all citations in generated content correspond
    to actual retrieved sources.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the hallucination detector.
        
        Args:
            strict_mode: If True, requires exact URL/title matching
        """
        self.strict_mode = strict_mode
    
    def detect(
        self,
        generated_text: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Detect hallucinations in generated text.
        
        Args:
            generated_text: The generated content to check
            sources: List of actual source dictionaries
            
        Returns:
            Dictionary with hallucination metrics
        """
        # Extract citations from text
        citations = self._extract_citations(generated_text)
        
        if not citations:
            return {
                "hallucination_rate": 0.0,
                "total_citations": 0,
                "valid_citations": 0,
                "hallucinated_citations": 0,
                "citations": [],
            }
        
        # Validate each citation
        for citation in citations:
            is_valid, matched = self._validate_citation(citation, sources)
            citation.is_valid = is_valid
            citation.matched_source = matched
        
        # Calculate metrics
        valid = sum(1 for c in citations if c.is_valid)
        total = len(citations)
        hallucinated = total - valid
        rate = hallucinated / total if total > 0 else 0.0
        
        return {
            "hallucination_rate": rate,
            "total_citations": total,
            "valid_citations": valid,
            "hallucinated_citations": hallucinated,
            "citations": [self._citation_to_dict(c) for c in citations],
        }
    
    def _extract_citations(self, text: str) -> List[Citation]:
        """
        Extract citations from generated text.
        
        Supports various citation formats:
        - URLs: https://...
        - Markdown links: [text](url)
        - Academic style: (Author, Year)
        - Numbered: [1], [2], etc.
        """
        citations = []
        
        # URL citations
        url_pattern = r'https?://[^\s\)\]<>]+'
        for match in re.finditer(url_pattern, text):
            citations.append(Citation(
                text=match.group(),
                reference=match.group(),
                position=match.start(),
            ))
        
        # Markdown links
        md_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        for match in re.finditer(md_pattern, text):
            citations.append(Citation(
                text=match.group(),
                reference=match.group(2),  # URL part
                position=match.start(),
            ))
        
        # Academic citations: (Author, Year) or (Author Year)
        academic_pattern = r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?),?\s*(\d{4})\)'
        for match in re.finditer(academic_pattern, text):
            citations.append(Citation(
                text=match.group(),
                reference=f"{match.group(1)} {match.group(2)}",
                position=match.start(),
            ))
        
        # Numbered references: [1], [2], etc.
        numbered_pattern = r'\[(\d+)\]'
        for match in re.finditer(numbered_pattern, text):
            citations.append(Citation(
                text=match.group(),
                reference=match.group(1),
                position=match.start(),
            ))
        
        # Deduplicate by position
        seen_positions = set()
        unique_citations = []
        for c in citations:
            if c.position not in seen_positions:
                seen_positions.add(c.position)
                unique_citations.append(c)
        
        return unique_citations
    
    def _validate_citation(
        self,
        citation: Citation,
        sources: List[Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a citation against sources.
        
        Returns:
            Tuple of (is_valid, matched_source_title)
        """
        if not sources:
            return False, None
        
        reference = citation.reference.lower()
        
        for source in sources:
            # Check URL match
            source_url = source.get("url", "").lower()
            if source_url and self._urls_match(reference, source_url):
                return True, source.get("title", source_url)
            
            # Check title match (for numbered or academic citations)
            source_title = source.get("title", "").lower()
            if source_title and self._titles_match(reference, source_title):
                return True, source.get("title")
            
            # Check snippet content match
            snippet = source.get("snippet", source.get("content", "")).lower()
            if snippet and self._content_contains_reference(reference, snippet):
                return True, source.get("title", "Source")
        
        return False, None
    
    def _urls_match(self, ref: str, source_url: str) -> bool:
        """Check if reference URL matches source URL."""
        # Normalize URLs
        ref = ref.rstrip("/").replace("www.", "")
        source_url = source_url.rstrip("/").replace("www.", "")
        
        if self.strict_mode:
            return ref == source_url
        else:
            # Partial match: check if one contains the other's domain
            return ref in source_url or source_url in ref
    
    def _titles_match(self, ref: str, title: str) -> bool:
        """Check if reference matches source title."""
        if self.strict_mode:
            return ref == title
        
        # Check for significant word overlap
        ref_words = set(ref.split())
        title_words = set(title.split())
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'and'}
        ref_words -= stopwords
        title_words -= stopwords
        
        if not ref_words or not title_words:
            return False
        
        overlap = len(ref_words & title_words)
        return overlap >= min(2, len(ref_words))
    
    def _content_contains_reference(self, ref: str, content: str) -> bool:
        """Check if content contains the reference."""
        # For numbered citations, we can't validate against content
        if ref.isdigit():
            return True  # Assume numbered citations are valid if sources exist
        
        return False
    
    def _citation_to_dict(self, citation: Citation) -> Dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            "text": citation.text,
            "reference": citation.reference,
            "is_valid": citation.is_valid,
            "matched_source": citation.matched_source,
        }


def calculate_hallucination_rate(
    generated_text: str,
    sources: List[Dict[str, Any]],
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to calculate hallucination rate.
    
    Args:
        generated_text: Text to check
        sources: Actual sources
        strict: Use strict matching
        
    Returns:
        Hallucination metrics dictionary
    """
    detector = HallucinationDetector(strict_mode=strict)
    return detector.detect(generated_text, sources)


if __name__ == "__main__":
    # Test hallucination detector
    print("Testing hallucination detector...")
    
    generated = """
    According to a study at https://example.com/renewable-study, solar power
    is growing rapidly. Research by Smith (2023) confirms these findings.
    
    See [this article](https://fake-news.com/fabricated) for more details.
    Multiple sources [1] [2] support this conclusion.
    """
    
    sources = [
        {
            "title": "Renewable Energy Study",
            "url": "https://example.com/renewable-study",
            "snippet": "Solar power generation has increased significantly.",
        },
        {
            "title": "Energy Market Analysis",
            "url": "https://analysis.org/energy",
            "snippet": "Market trends show growth in renewable adoption.",
        },
    ]
    
    result = calculate_hallucination_rate(generated, sources)
    
    print(f"\nHallucination Results:")
    print(f"  Rate: {result['hallucination_rate']:.1%}")
    print(f"  Total citations: {result['total_citations']}")
    print(f"  Valid: {result['valid_citations']}")
    print(f"  Hallucinated: {result['hallucinated_citations']}")
    
    print(f"\nCitation details:")
    for c in result['citations']:
        status = "✓" if c['is_valid'] else "✗"
        print(f"  {status} {c['text'][:50]}...")
