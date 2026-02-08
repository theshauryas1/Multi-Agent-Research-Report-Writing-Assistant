"""
Writer Agent - Writes report sections based on outline and research.
"""

from typing import Any, Dict, List

from .base_agent import BaseAgent
from config import PROMPTS


class WriterAgent(BaseAgent):
    """
    Agent responsible for writing report content.
    
    This agent:
    1. Takes the outline and research as context
    2. Writes each section sequentially
    3. Maintains coherence across sections
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="Writer Agent",
            agent_type="writer",
            verbose=verbose,
        )
    
    def run(
        self,
        topic: str,
        outline: List[Dict[str, str]],
        research_summary: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Write all sections of the report.
        
        Args:
            topic: The main topic
            outline: List of section dictionaries with 'title' and 'description'
            research_summary: Research context to use
            
        Returns:
            Dictionary containing:
                - sections: List of written sections
                - full_draft: Complete draft as a single string
        """
        self.log(f"Starting to write report on: '{topic}'")
        self.log(f"Writing {len(outline)} sections...")
        
        sections = []
        previous_content = ""
        
        for i, section_info in enumerate(outline):
            section_title = section_info["title"]
            section_desc = section_info["description"]
            
            self.log(f"Writing section {i+1}/{len(outline)}: {section_title}")
            
            # Write the section
            section_content = self._write_section(
                topic=topic,
                section_title=section_title,
                section_description=section_desc,
                research=research_summary,
                previous_sections=previous_content,
            )
            
            sections.append({
                "title": section_title,
                "description": section_desc,
                "content": section_content,
            })
            
            # Update previous content for context
            previous_content += f"\n\n## {section_title}\n{section_content}"
        
        # Compile full draft
        full_draft = self._compile_draft(topic, sections)
        
        self.log("Draft complete!")
        
        return {
            "sections": sections,
            "full_draft": full_draft,
            "topic": topic,
        }
    
    def write_single_section(
        self,
        topic: str,
        section_title: str,
        section_description: str,
        research_summary: str,
        previous_sections: str = "",
    ) -> str:
        """
        Write a single section (useful for rewrites).
        
        Args:
            topic: The main topic
            section_title: Title of the section to write
            section_description: What the section should cover
            research_summary: Research context
            previous_sections: Content of previous sections for context
            
        Returns:
            The written section content
        """
        self.log(f"Writing section: {section_title}")
        
        return self._write_section(
            topic=topic,
            section_title=section_title,
            section_description=section_description,
            research=research_summary,
            previous_sections=previous_sections,
        )
    
    def _write_section(
        self,
        topic: str,
        section_title: str,
        section_description: str,
        research: str,
        previous_sections: str,
    ) -> str:
        """Internal method to write a single section."""
        
        prompt = PROMPTS["writer"].format(
            topic=topic,
            section_title=section_title,
            section_description=section_description,
            research=research[:3000],  # Limit research context
            previous_sections=previous_sections[-2000:] if previous_sections else "This is the first section.",
        )
        
        try:
            content = self.invoke_llm(prompt)
            return content.strip()
        except Exception as e:
            self.log(f"Failed to write section '{section_title}': {e}", "error")
            # Return a placeholder
            return self._fallback_section(section_title, section_description)
    
    def _fallback_section(self, title: str, description: str) -> str:
        """Generate fallback content when LLM fails."""
        return f"""
{description}

This section discusses the key aspects of {title.lower()}. Due to technical limitations, 
detailed content generation was not possible. Please consider expanding this section 
with relevant information from the research materials.
""".strip()
    
    def _compile_draft(self, topic: str, sections: List[Dict]) -> str:
        """Compile all sections into a complete draft."""
        draft = f"# {topic}\n\n"
        
        for section in sections:
            draft += f"## {section['title']}\n\n"
            draft += f"{section['content']}\n\n"
        
        return draft.strip()


if __name__ == "__main__":
    # Test the writer agent
    agent = WriterAgent()
    
    test_outline = [
        {"title": "Introduction", "description": "Overview of renewable energy"},
        {"title": "Benefits", "description": "Key advantages of renewable energy"},
    ]
    
    test_research = "Renewable energy includes solar, wind, and hydro power..."
    
    result = agent.run(
        topic="Renewable Energy",
        outline=test_outline,
        research_summary=test_research,
    )
    
    print("\n" + "="*50)
    print("WRITING RESULTS")
    print("="*50)
    print(f"\nSections written: {len(result['sections'])}")
    print("\nDraft preview:")
    print(result['full_draft'][:500] + "...")
