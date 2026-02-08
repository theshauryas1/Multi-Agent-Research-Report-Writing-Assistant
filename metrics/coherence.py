"""
Coherence Scorer - Measures structural coherence and redundancy.
Evaluates logical flow, topic consistency, and content overlap.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class CoherenceScorer:
    """
    Scores structural coherence of generated reports.
    
    Evaluates:
    - Section-to-section flow
    - Topic consistency
    - Redundancy between sections
    """
    
    def __init__(
        self,
        use_semantic: bool = True,
        redundancy_threshold: float = 0.7,
    ):
        """
        Initialize the coherence scorer.
        
        Args:
            use_semantic: Use semantic embeddings for scoring
            redundancy_threshold: Similarity above this is redundant
        """
        self.use_semantic = use_semantic
        self.redundancy_threshold = redundancy_threshold
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and self.use_semantic:
            try:
                from rag.embeddings import get_embedding_model
                self._embedding_model = get_embedding_model()
            except ImportError:
                print("Warning: Could not load embedding model")
                self.use_semantic = False
        return self._embedding_model
    
    def score(
        self,
        sections: List[Dict[str, str]],
        outline: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Score the coherence of report sections.
        
        Args:
            sections: List of sections with 'title' and 'content'
            outline: Optional outline for structure validation
            
        Returns:
            Dictionary with coherence metrics
        """
        if not sections:
            return {
                "overall_score": 1.0,
                "section_scores": [],
                "redundancy_rate": 0.0,
                "topic_drift": 0.0,
            }
        
        # Calculate individual metrics
        section_scores = self._score_sections(sections)
        flow_score = self._score_flow(sections)
        redundancy = self._calculate_redundancy(sections)
        topic_drift = self._calculate_topic_drift(sections)
        
        # Structure score (if outline provided)
        structure_score = 1.0
        if outline:
            structure_score = self._score_structure(sections, outline)
        
        # Combine into overall score
        overall = (
            0.3 * np.mean(section_scores) +
            0.25 * flow_score +
            0.25 * (1.0 - redundancy["rate"]) +
            0.1 * (1.0 - topic_drift) +
            0.1 * structure_score
        )
        
        return {
            "overall_score": float(overall),
            "section_scores": section_scores,
            "flow_score": float(flow_score),
            "redundancy_rate": redundancy["rate"],
            "avg_section_similarity": redundancy["avg_similarity"],
            "topic_drift": float(topic_drift),
            "structure_score": float(structure_score),
        }
    
    def _score_sections(self, sections: List[Dict[str, str]]) -> List[float]:
        """Score individual sections based on quality heuristics."""
        scores = []
        
        for section in sections:
            score = self._score_single_section(section)
            scores.append(score)
        
        return scores
    
    def _score_single_section(self, section: Dict[str, str]) -> float:
        """Score a single section."""
        content = section.get("content", "")
        title = section.get("title", "")
        
        score = 0.0
        
        # Length score (300-800 words ideal)
        words = len(content.split())
        if words >= 300:
            score += 0.3
        elif words >= 150:
            score += 0.2
        elif words >= 50:
            score += 0.1
        
        if words <= 800:
            score += 0.1
        
        # Structure score (paragraphs, not walls of text)
        paragraphs = content.split("\n\n")
        if 2 <= len(paragraphs) <= 10:
            score += 0.2
        elif len(paragraphs) >= 1:
            score += 0.1
        
        # Title relevance (check if title words appear in content)
        title_words = set(title.lower().split())
        content_lower = content.lower()
        title_coverage = sum(1 for w in title_words if w in content_lower)
        if title_words:
            score += 0.2 * (title_coverage / len(title_words))
        
        # Sentence variety (avoid repetitive patterns)
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 3:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                variance = np.var(lengths) if len(lengths) > 1 else 0
                if variance > 10:  # Good variety
                    score += 0.1
        
        # No obvious issues
        has_issues = False
        if "[" in content and "]" in content and "(" not in content:
            has_issues = True  # Incomplete markdown
        if content.count("...") > 3:
            has_issues = True  # Too many ellipses
        
        if not has_issues:
            score += 0.1
        
        return min(1.0, score)
    
    def _score_flow(self, sections: List[Dict[str, str]]) -> float:
        """Score the flow between consecutive sections."""
        if len(sections) < 2:
            return 1.0
        
        if not self.use_semantic or not self.embedding_model:
            return self._score_flow_keywords(sections)
        
        # Get embeddings for each section
        contents = [s.get("content", "") for s in sections]
        embeddings = self.embedding_model.embed(contents)
        
        # Calculate consecutive similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Good flow = moderate similarity (not too high, not too low)
        # Ideal range: 0.3 - 0.7
        flow_scores = []
        for sim in similarities:
            if 0.3 <= sim <= 0.7:
                flow_scores.append(1.0)
            elif 0.2 <= sim <= 0.8:
                flow_scores.append(0.8)
            elif 0.1 <= sim <= 0.9:
                flow_scores.append(0.6)
            else:
                flow_scores.append(0.4)
        
        return np.mean(flow_scores) if flow_scores else 1.0
    
    def _score_flow_keywords(self, sections: List[Dict[str, str]]) -> float:
        """Score flow using keyword overlap (fallback)."""
        if len(sections) < 2:
            return 1.0
        
        scores = []
        for i in range(len(sections) - 1):
            current = set(sections[i].get("content", "").lower().split())
            next_sec = set(sections[i + 1].get("content", "").lower().split())
            
            if current and next_sec:
                overlap = len(current & next_sec) / len(current | next_sec)
                # Ideal overlap: 10-30%
                if 0.1 <= overlap <= 0.3:
                    scores.append(1.0)
                elif 0.05 <= overlap <= 0.4:
                    scores.append(0.8)
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        
        return np.mean(scores) if scores else 1.0
    
    def _calculate_redundancy(
        self, 
        sections: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """Calculate redundancy between sections."""
        if len(sections) < 2:
            return {"rate": 0.0, "avg_similarity": 0.0}
        
        contents = [s.get("content", "") for s in sections]
        
        if self.use_semantic and self.embedding_model:
            embeddings = self.embedding_model.embed(contents)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    similarities.append(sim)
            
            avg_sim = np.mean(similarities) if similarities else 0.0
            redundant = sum(1 for s in similarities if s > self.redundancy_threshold)
            rate = redundant / len(similarities) if similarities else 0.0
            
            return {"rate": rate, "avg_similarity": avg_sim}
        else:
            # Keyword-based redundancy
            return self._calculate_redundancy_keywords(contents)
    
    def _calculate_redundancy_keywords(
        self, 
        contents: List[str],
    ) -> Dict[str, float]:
        """Calculate redundancy using n-gram overlap."""
        if len(contents) < 2:
            return {"rate": 0.0, "avg_similarity": 0.0}
        
        # Extract 3-grams
        def get_ngrams(text: str, n: int = 3) -> set:
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        
        ngrams = [get_ngrams(c) for c in contents]
        
        overlaps = []
        for i in range(len(ngrams)):
            for j in range(i + 1, len(ngrams)):
                if ngrams[i] and ngrams[j]:
                    overlap = len(ngrams[i] & ngrams[j])
                    total = len(ngrams[i] | ngrams[j])
                    overlaps.append(overlap / total if total > 0 else 0.0)
        
        if not overlaps:
            return {"rate": 0.0, "avg_similarity": 0.0}
        
        avg_overlap = np.mean(overlaps)
        redundant = sum(1 for o in overlaps if o > 0.3)
        rate = redundant / len(overlaps)
        
        return {"rate": rate, "avg_similarity": avg_overlap}
    
    def _calculate_topic_drift(self, sections: List[Dict[str, str]]) -> float:
        """Calculate topic drift from first to last section."""
        if len(sections) < 2:
            return 0.0
        
        first = sections[0].get("content", "")
        last = sections[-1].get("content", "")
        
        if self.use_semantic and self.embedding_model:
            embeddings = self.embedding_model.embed([first, last])
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            # High similarity = low drift
            return 1.0 - similarity
        else:
            # Keyword overlap
            first_words = set(first.lower().split())
            last_words = set(last.lower().split())
            
            if not first_words or not last_words:
                return 0.5
            
            overlap = len(first_words & last_words) / len(first_words | last_words)
            return 1.0 - overlap
    
    def _score_structure(
        self,
        sections: List[Dict[str, str]],
        outline: List[Dict[str, str]],
    ) -> float:
        """Score how well sections follow the outline."""
        if not outline:
            return 1.0
        
        # Check title matching
        outline_titles = [o.get("title", "").lower() for o in outline]
        section_titles = [s.get("title", "").lower() for s in sections]
        
        matches = 0
        for i, outline_title in enumerate(outline_titles):
            if i < len(section_titles):
                # Fuzzy match
                section_title = section_titles[i]
                outline_words = set(outline_title.split())
                section_words = set(section_title.split())
                
                if outline_words & section_words:
                    matches += 1
        
        return matches / len(outline_titles) if outline_titles else 1.0


def calculate_coherence_score(
    sections: List[Dict[str, str]],
    outline: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to calculate coherence score.
    
    Args:
        sections: Report sections
        outline: Optional outline
        
    Returns:
        Coherence metrics dictionary
    """
    scorer = CoherenceScorer()
    return scorer.score(sections, outline)


if __name__ == "__main__":
    # Test coherence scorer
    print("Testing coherence scorer...")
    
    sections = [
        {
            "title": "Introduction",
            "content": """
            Renewable energy is becoming increasingly important in the global effort
            to combat climate change. This report explores the benefits and challenges
            of transitioning to clean energy sources. We will examine solar, wind,
            and hydroelectric power as key alternatives to fossil fuels.
            """,
        },
        {
            "title": "Solar Energy",
            "content": """
            Solar power harnesses energy from sunlight using photovoltaic cells.
            The technology has improved dramatically, with efficiency rates now
            exceeding 20% for commercial panels. Solar installations have grown
            by 40% annually over the past decade.
            """,
        },
        {
            "title": "Wind Energy", 
            "content": """
            Wind turbines convert kinetic energy from moving air into electricity.
            Modern wind farms can generate power at costs competitive with fossil
            fuels. Offshore wind installations offer even greater potential due
            to stronger, more consistent winds.
            """,
        },
        {
            "title": "Conclusion",
            "content": """
            The transition to renewable energy is essential for a sustainable future.
            Solar and wind power offer viable alternatives to fossil fuels, with
            improving technology and falling costs. Investment in clean energy
            infrastructure should be a priority for all nations.
            """,
        },
    ]
    
    result = calculate_coherence_score(sections)
    
    print(f"\nCoherence Results:")
    print(f"  Overall score: {result['overall_score']:.2f}")
    print(f"  Flow score: {result['flow_score']:.2f}")
    print(f"  Redundancy rate: {result['redundancy_rate']:.1%}")
    print(f"  Topic drift: {result['topic_drift']:.2f}")
    print(f"\nSection scores: {result['section_scores']}")
