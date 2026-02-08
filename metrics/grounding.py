"""
Grounding Evaluator - Measures claim-to-source grounding accuracy.
Extracts claims from generated text and validates against retrieved sources.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Claim:
    """A factual claim extracted from generated text."""
    text: str
    section: str
    sentence_index: int
    is_grounded: bool = False
    matching_source: Optional[str] = None
    confidence: float = 0.0


class GroundingEvaluator:
    """
    Evaluates claim-to-source grounding accuracy.
    
    Extracts factual claims from generated content and validates
    them against retrieved source chunks.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        use_semantic_matching: bool = True,
    ):
        """
        Initialize the grounding evaluator.
        
        Args:
            similarity_threshold: Minimum similarity for a match
            use_semantic_matching: Use embeddings for matching
        """
        self.similarity_threshold = similarity_threshold
        self.use_semantic_matching = use_semantic_matching
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and self.use_semantic_matching:
            try:
                from rag.embeddings import get_embedding_model
                self._embedding_model = get_embedding_model()
            except ImportError:
                print("Warning: Could not load embedding model, falling back to keyword matching")
                self.use_semantic_matching = False
        return self._embedding_model
    
    def evaluate(
        self,
        generated_text: str,
        sources: List[Dict[str, Any]],
        section_name: str = "content",
    ) -> Dict[str, Any]:
        """
        Evaluate grounding of generated text against sources.
        
        Args:
            generated_text: The generated content to evaluate
            sources: List of source chunks with 'content' field
            section_name: Name of the section being evaluated
            
        Returns:
            Dictionary with grounding metrics
        """
        # Extract claims
        claims = self._extract_claims(generated_text, section_name)
        
        if not claims:
            return {
                "accuracy": 1.0,  # No claims = technically all grounded
                "total_claims": 0,
                "grounded_claims": 0,
                "ungrounded_claims": 0,
                "claims": [],
            }
        
        # Validate each claim
        source_texts = [s.get("content", s.get("snippet", "")) for s in sources]
        
        for claim in claims:
            is_grounded, source, confidence = self._validate_claim(
                claim.text, source_texts
            )
            claim.is_grounded = is_grounded
            claim.matching_source = source
            claim.confidence = confidence
        
        # Calculate metrics
        grounded = sum(1 for c in claims if c.is_grounded)
        total = len(claims)
        accuracy = grounded / total if total > 0 else 1.0
        
        return {
            "accuracy": accuracy,
            "total_claims": total,
            "grounded_claims": grounded,
            "ungrounded_claims": total - grounded,
            "claims": [self._claim_to_dict(c) for c in claims],
        }
    
    def _extract_claims(
        self, 
        text: str, 
        section: str,
    ) -> List[Claim]:
        """
        Extract factual claims from text.
        
        Focuses on sentences that make factual assertions.
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            
            if not sentence or len(sentence) < 20:
                continue
            
            # Identify factual claims (heuristic-based)
            if self._is_factual_claim(sentence):
                claims.append(Claim(
                    text=sentence,
                    section=section,
                    sentence_index=i,
                ))
        
        return claims
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Determine if a sentence makes a factual claim.
        
        Uses heuristics to identify assertive statements.
        """
        # Skip questions
        if sentence.endswith("?"):
            return False
        
        # Skip very short sentences
        if len(sentence.split()) < 5:
            return False
        
        # Look for factual indicators
        factual_patterns = [
            r'\d+%',  # Percentages
            r'\d+\s*(million|billion|thousand)',  # Numbers
            r'(study|research|data|evidence|report)\s+shows?',
            r'according to',
            r'(is|are|was|were)\s+(known|recognized|considered|estimated)',
            r'(increase|decrease|growth|decline)\s+of',
            r'(most|many|some|all|few)\s+\w+\s+(are|is|have|has)',
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Default: treat descriptive sentences as claims
        descriptive_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'includes', 'contains']
        words = sentence.lower().split()
        
        for verb in descriptive_verbs:
            if verb in words[:10]:  # Check early in sentence
                return True
        
        return False
    
    def _validate_claim(
        self,
        claim: str,
        sources: List[str],
    ) -> Tuple[bool, Optional[str], float]:
        """
        Validate a claim against source texts.
        
        Returns:
            Tuple of (is_grounded, matching_source, confidence)
        """
        if not sources:
            return False, None, 0.0
        
        if self.use_semantic_matching and self.embedding_model:
            return self._semantic_match(claim, sources)
        else:
            return self._keyword_match(claim, sources)
    
    def _semantic_match(
        self,
        claim: str,
        sources: List[str],
    ) -> Tuple[bool, Optional[str], float]:
        """Match claim using semantic similarity."""
        import numpy as np
        
        # Embed claim and sources
        claim_embedding = self.embedding_model.embed_query(claim)
        source_embeddings = self.embedding_model.embed(sources)
        
        # Calculate similarities
        similarities = np.dot(source_embeddings, claim_embedding)
        
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        
        is_grounded = best_score >= self.similarity_threshold
        matching_source = sources[best_idx] if is_grounded else None
        
        return is_grounded, matching_source, best_score
    
    def _keyword_match(
        self,
        claim: str,
        sources: List[str],
    ) -> Tuple[bool, Optional[str], float]:
        """Match claim using keyword overlap."""
        # Extract meaningful words from claim
        claim_words = set(self._extract_keywords(claim))
        
        if not claim_words:
            return False, None, 0.0
        
        best_score = 0.0
        best_source = None
        
        for source in sources:
            source_words = set(self._extract_keywords(source))
            
            if not source_words:
                continue
            
            # Jaccard similarity
            overlap = len(claim_words & source_words)
            union = len(claim_words | source_words)
            score = overlap / union if union > 0 else 0.0
            
            if score > best_score:
                best_score = score
                best_source = source
        
        is_grounded = best_score >= self.similarity_threshold * 0.5  # Lower threshold for keywords
        
        return is_grounded, best_source, best_score
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'it', 'its', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]
        
        return keywords
    
    def _claim_to_dict(self, claim: Claim) -> Dict[str, Any]:
        """Convert claim to dictionary."""
        return {
            "text": claim.text,
            "section": claim.section,
            "is_grounded": claim.is_grounded,
            "confidence": claim.confidence,
            "matching_source": claim.matching_source[:200] if claim.matching_source else None,
        }


def calculate_grounding_accuracy(
    generated_text: str,
    sources: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Convenience function to calculate grounding accuracy.
    
    Args:
        generated_text: Text to evaluate
        sources: Source chunks
        threshold: Similarity threshold
        
    Returns:
        Grounding metrics dictionary
    """
    evaluator = GroundingEvaluator(similarity_threshold=threshold)
    return evaluator.evaluate(generated_text, sources)


if __name__ == "__main__":
    # Test grounding evaluator
    print("Testing grounding evaluator...")
    
    generated = """
    Renewable energy sources now account for 30% of global electricity generation.
    Solar power has seen a 50% cost reduction over the past decade.
    Wind energy is the fastest-growing renewable source in many countries.
    The transition to clean energy is essential for combating climate change.
    """
    
    sources = [
        {"content": "According to recent data, renewable energy makes up approximately 30% of global power generation."},
        {"content": "Solar panel costs have dropped by 50% since 2010, making solar power more accessible."},
        {"content": "Wind power capacity has grown significantly, becoming a leading renewable source."},
    ]
    
    result = calculate_grounding_accuracy(generated, sources)
    
    print(f"\nGrounding Results:")
    print(f"  Accuracy: {result['accuracy']:.1%}")
    print(f"  Total claims: {result['total_claims']}")
    print(f"  Grounded: {result['grounded_claims']}")
    print(f"  Ungrounded: {result['ungrounded_claims']}")
