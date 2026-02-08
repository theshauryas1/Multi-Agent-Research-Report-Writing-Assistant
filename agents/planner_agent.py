"""
Planner Agent - Creates structured outlines for reports.
"""

from typing import Any, Dict, List
import re
import json

from .base_agent import BaseAgent
from config import PROMPTS


class PlannerAgent(BaseAgent):
    """
    Agent responsible for creating report outlines.
    
    This agent:
    1. Analyzes the research summary
    2. Identifies key themes and topics
    3. Creates a logical structure for the report
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="Planner Agent",
            agent_type="default",
            verbose=verbose,
        )
    
    def run(
        self,
        topic: str,
        research_summary: str,
        num_sections: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a report outline based on research.
        
        Args:
            topic: The main topic
            research_summary: Summary from research agent
            num_sections: Target number of main sections
            
        Returns:
            Dictionary containing:
                - outline: List of section dictionaries
                - outline_text: Text representation of the outline
        """
        self.log(f"Planning report structure for: '{topic}'")
        
        # Build the planning prompt
        prompt = PROMPTS["planner"].format(
            topic=topic,
            research=research_summary,
        )
        prompt += f"\n\nAim for approximately {num_sections} main sections."
        prompt += """

Format your outline as follows:
1. [Section Title]
   Description: [What this section should cover]

2. [Section Title]
   Description: [What this section should cover]

...and so on.
"""
        
        self.log("Generating outline with LLM...")
        
        try:
            outline_text = self.invoke_llm(prompt)
            outline = self._parse_outline(outline_text)
        except Exception as e:
            self.log(f"LLM outline generation failed, using default: {e}", "warning")
            outline = self._default_outline(topic)
            outline_text = self._outline_to_text(outline)
        
        self.log(f"Created outline with {len(outline)} sections")
        
        return {
            "outline": outline,
            "outline_text": outline_text,
            "topic": topic,
        }
    
    def _parse_outline(self, outline_text: str) -> List[Dict[str, str]]:
        """
        Parse the LLM's outline text into a structured format.
        
        Returns a list of dictionaries with 'title' and 'description' keys.
        """
        outline = []
        
        # Pattern to match numbered sections
        # Matches lines like "1. Section Title" or "1. **Section Title**"
        section_pattern = r'^\d+\.\s*\*{0,2}(.+?)\*{0,2}\s*$'
        desc_pattern = r'^\s*Description:\s*(.+)$'
        
        lines = outline_text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Check for section header
            section_match = re.match(section_pattern, line)
            if section_match:
                # Save previous section if exists
                if current_section:
                    outline.append(current_section)
                
                current_section = {
                    "title": section_match.group(1).strip(),
                    "description": "",
                }
                continue
            
            # Check for description
            desc_match = re.match(desc_pattern, line, re.IGNORECASE)
            if desc_match and current_section:
                current_section["description"] = desc_match.group(1).strip()
                continue
            
            # If line has content and we have a current section, append to description
            if line and current_section and not current_section["description"]:
                # Check if it's a continuation or bullets
                if line.startswith('-') or line.startswith('•'):
                    current_section["description"] += line + " "
                elif not re.match(r'^\d+\.', line):
                    current_section["description"] = line
        
        # Don't forget the last section
        if current_section:
            outline.append(current_section)
        
        # If parsing failed, create a basic outline
        if not outline:
            outline = self._default_outline("the topic")
        
        return outline
    
    def _default_outline(self, topic: str) -> List[Dict[str, str]]:
        """Generate a default outline structure."""
        return [
            {
                "title": "Introduction",
                "description": f"An overview of {topic} and why it matters."
            },
            {
                "title": "Background and Context",
                "description": f"Historical context and foundational concepts related to {topic}."
            },
            {
                "title": "Key Concepts and Components",
                "description": f"The main elements and principles of {topic}."
            },
            {
                "title": "Current Applications and Impact",
                "description": f"How {topic} is being applied today and its effects."
            },
            {
                "title": "Challenges and Considerations",
                "description": f"Current challenges and important considerations regarding {topic}."
            },
            {
                "title": "Future Directions",
                "description": f"Emerging trends and future outlook for {topic}."
            },
            {
                "title": "Conclusion",
                "description": f"Summary of key points and final thoughts on {topic}."
            },
        ]
    
    def _outline_to_text(self, outline: List[Dict[str, str]]) -> str:
        """Convert structured outline to text format."""
        text = "# Report Outline\n\n"
        
        for i, section in enumerate(outline, 1):
            text += f"{i}. **{section['title']}**\n"
            text += f"   {section['description']}\n\n"
        
        return text


if __name__ == "__main__":
    # Test the planner agent
    agent = PlannerAgent()
    
    test_research = """
    Renewable energy is energy derived from natural sources that are replenished at a higher rate 
    than they are consumed. Solar, wind, hydroelectric, and geothermal are the main types.
    Benefits include reduced greenhouse gas emissions, energy independence, and job creation.
    Challenges include intermittency, storage solutions, and initial infrastructure costs.
    """
    
    result = agent.run(
        topic="Benefits of Renewable Energy",
        research_summary=test_research,
    )
    
    print("\n" + "="*50)
    print("PLANNING RESULTS")
    print("="*50)
    print(f"\nSections planned: {len(result['outline'])}")
    print("\nOutline:")
    for section in result['outline']:
        print(f"  - {section['title']}")
