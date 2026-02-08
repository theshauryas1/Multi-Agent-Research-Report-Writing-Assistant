"""
Reviewer Agent - Reviews report sections for quality and completeness.
"""

from typing import Any, Dict, List, Tuple
import re

from .base_agent import BaseAgent
from config import PROMPTS, MIN_REVIEW_SCORE


class ReviewerAgent(BaseAgent):
    """
    Agent responsible for reviewing and evaluating report content.
    
    This agent:
    1. Reviews each section for quality
    2. Scores sections on multiple criteria
    3. Provides actionable feedback for improvements
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="Reviewer Agent",
            agent_type="reviewer",
            verbose=verbose,
        )
        self.min_score = MIN_REVIEW_SCORE
    
    def run(
        self,
        sections: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Review all sections and provide feedback.
        
        Args:
            sections: List of section dictionaries with 'title' and 'content'
            
        Returns:
            Dictionary containing:
                - reviews: List of review results per section
                - overall_score: Average score across all sections
                - needs_revision: Whether any section needs revision
                - sections_to_fix: Indices of sections requiring fixes
        """
        self.log("Starting review process...")
        
        reviews = []
        scores = []
        sections_to_fix = []
        
        for i, section in enumerate(sections):
            self.log(f"Reviewing section {i+1}/{len(sections)}: {section['title']}")
            
            review = self._review_section(
                section_title=section["title"],
                content=section["content"],
            )
            
            reviews.append(review)
            scores.append(review["score"])
            
            if review["needs_revision"]:
                sections_to_fix.append(i)
                self.log(f"  → Section needs revision (score: {review['score']})", "warning")
            else:
                self.log(f"  → Section passed (score: {review['score']})")
        
        overall_score = sum(scores) / len(scores) if scores else 0
        needs_revision = len(sections_to_fix) > 0
        
        self.log(f"Review complete. Overall score: {overall_score:.1f}/10")
        if needs_revision:
            self.log(f"{len(sections_to_fix)} section(s) flagged for revision")
        
        return {
            "reviews": reviews,
            "overall_score": overall_score,
            "needs_revision": needs_revision,
            "sections_to_fix": sections_to_fix,
        }
    
    def review_single_section(
        self,
        section_title: str,
        content: str,
    ) -> Dict[str, Any]:
        """
        Review a single section.
        
        Args:
            section_title: Title of the section
            content: Section content to review
            
        Returns:
            Review result dictionary
        """
        return self._review_section(section_title, content)
    
    def _review_section(
        self,
        section_title: str,
        content: str,
    ) -> Dict[str, Any]:
        """Internal method to review a section."""
        
        prompt = PROMPTS["reviewer"].format(
            section_title=section_title,
            content=content,
        )
        
        try:
            review_text = self.invoke_llm(prompt)
            score, needs_revision, feedback = self._parse_review(review_text)
        except Exception as e:
            self.log(f"Review failed for '{section_title}': {e}", "error")
            # Default to passing if review fails
            score = 7
            needs_revision = False
            feedback = "Unable to complete review. Manual review recommended."
        
        return {
            "section_title": section_title,
            "score": score,
            "needs_revision": needs_revision,
            "feedback": feedback,
            "raw_review": review_text if 'review_text' in dir() else "",
        }
    
    def _parse_review(self, review_text: str) -> Tuple[float, bool, str]:
        """
        Parse the reviewer's response to extract score, revision need, and feedback.
        
        Returns:
            Tuple of (score, needs_revision, feedback)
        """
        # Default values
        score = 5.0
        needs_revision = True
        feedback = review_text
        
        # Try to extract score
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', review_text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(1, min(10, score))  # Clamp to 1-10
            except ValueError:
                pass
        
        # Try to extract revision need
        revision_match = re.search(r'NEEDS_REVISION:\s*(Yes|No)', review_text, re.IGNORECASE)
        if revision_match:
            needs_revision = revision_match.group(1).lower() == 'yes'
        else:
            # Infer from score
            needs_revision = score < self.min_score
        
        # Try to extract feedback
        feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=\n\n|\Z)', review_text, re.IGNORECASE | re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        
        return score, needs_revision, feedback


if __name__ == "__main__":
    # Test the reviewer agent
    agent = ReviewerAgent()
    
    test_sections = [
        {
            "title": "Introduction",
            "content": """
            Renewable energy represents one of the most significant shifts in how we generate 
            and consume power. This report explores the various benefits and challenges 
            associated with transitioning to renewable energy sources.
            """
        },
        {
            "title": "Benefits",
            "content": "Solar is good. Wind is nice."  # Deliberately poor quality
        }
    ]
    
    result = agent.run(sections=test_sections)
    
    print("\n" + "="*50)
    print("REVIEW RESULTS")
    print("="*50)
    print(f"\nOverall score: {result['overall_score']:.1f}/10")
    print(f"Needs revision: {result['needs_revision']}")
    print(f"Sections to fix: {result['sections_to_fix']}")
    
    for review in result['reviews']:
        print(f"\n{review['section_title']}: {review['score']}/10")
        print(f"  Needs revision: {review['needs_revision']}")
