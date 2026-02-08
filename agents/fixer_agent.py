"""
Fixer Agent - Revises sections based on reviewer feedback.
"""

from typing import Any, Dict, List

from .base_agent import BaseAgent
from config import PROMPTS


class FixerAgent(BaseAgent):
    """
    Agent responsible for revising and improving flagged sections.
    
    This agent:
    1. Takes original content and review feedback
    2. Rewrites sections to address feedback
    3. Maintains the original intent while improving quality
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="Fixer Agent",
            agent_type="writer",  # Uses writer-type LLM for quality
            verbose=verbose,
        )
    
    def run(
        self,
        sections: List[Dict[str, str]],
        reviews: List[Dict[str, Any]],
        sections_to_fix: List[int],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fix flagged sections based on reviewer feedback.
        
        Args:
            sections: List of all section dictionaries
            reviews: List of review results
            sections_to_fix: Indices of sections that need fixing
            
        Returns:
            Dictionary containing:
                - fixed_sections: Updated sections list with fixes applied
                - changes_made: List of changes that were made
        """
        self.log(f"Starting fixes for {len(sections_to_fix)} section(s)...")
        
        # Make a copy to avoid modifying original
        fixed_sections = [s.copy() for s in sections]
        changes_made = []
        
        for idx in sections_to_fix:
            section = sections[idx]
            review = reviews[idx]
            
            self.log(f"Fixing section: {section['title']}")
            
            # Get the fixed content
            fixed_content = self._fix_section(
                section_title=section["title"],
                original_content=section["content"],
                feedback=review["feedback"],
            )
            
            # Update the section
            fixed_sections[idx]["content"] = fixed_content
            fixed_sections[idx]["was_revised"] = True
            
            changes_made.append({
                "section_index": idx,
                "section_title": section["title"],
                "original_length": len(section["content"]),
                "revised_length": len(fixed_content),
                "feedback_addressed": review["feedback"][:200],
            })
            
            self.log(f"  → Fixed (original: {len(section['content'])} chars, revised: {len(fixed_content)} chars)")
        
        self.log("All fixes complete!")
        
        return {
            "fixed_sections": fixed_sections,
            "changes_made": changes_made,
        }
    
    def fix_single_section(
        self,
        section_title: str,
        original_content: str,
        feedback: str,
    ) -> str:
        """
        Fix a single section based on feedback.
        
        Args:
            section_title: Title of the section
            original_content: Original section content
            feedback: Review feedback to address
            
        Returns:
            Revised section content
        """
        return self._fix_section(section_title, original_content, feedback)
    
    def _fix_section(
        self,
        section_title: str,
        original_content: str,
        feedback: str,
    ) -> str:
        """Internal method to fix a section."""
        
        prompt = PROMPTS["fixer"].format(
            original_content=original_content,
            feedback=feedback,
        )
        
        prompt = f"Section: {section_title}\n\n" + prompt
        
        try:
            fixed_content = self.invoke_llm(prompt)
            
            # Basic validation - ensure we got meaningful content
            if len(fixed_content.strip()) < 50:
                self.log(f"Fix too short, keeping original", "warning")
                return original_content
            
            return fixed_content.strip()
            
        except Exception as e:
            self.log(f"Failed to fix section '{section_title}': {e}", "error")
            # Return original if fix fails
            return original_content
    
    def compile_final_draft(
        self,
        topic: str,
        sections: List[Dict[str, str]],
    ) -> str:
        """
        Compile fixed sections into a final draft.
        
        Args:
            topic: The main topic
            sections: List of (potentially fixed) sections
            
        Returns:
            Complete final draft as a string
        """
        draft = f"# {topic}\n\n"
        
        for section in sections:
            draft += f"## {section['title']}\n\n"
            draft += f"{section['content']}\n\n"
            
        return draft.strip()


if __name__ == "__main__":
    # Test the fixer agent
    agent = FixerAgent()
    
    test_sections = [
        {
            "title": "Introduction",
            "content": "This is the intro. It's very short."
        },
        {
            "title": "Details",
            "content": "Some details here that need work."
        }
    ]
    
    test_reviews = [
        {"feedback": "Too short. Needs more context and depth.", "score": 4},
        {"feedback": "Lacks specific examples and data.", "score": 5},
    ]
    
    result = agent.run(
        sections=test_sections,
        reviews=test_reviews,
        sections_to_fix=[0, 1],
    )
    
    print("\n" + "="*50)
    print("FIXER RESULTS")
    print("="*50)
    print(f"\nChanges made: {len(result['changes_made'])}")
    
    for change in result['changes_made']:
        print(f"\n{change['section_title']}:")
        print(f"  Original: {change['original_length']} chars")
        print(f"  Revised: {change['revised_length']} chars")
