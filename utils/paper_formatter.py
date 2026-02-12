"""
Paper Formatter
Enforces strict paper structure and formatting
"""

from typing import List, Dict, Optional


# Required paper sections in order
PAPER_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Methodology",
    "Results",
    "Discussion",
    "Conclusion",
    "References"
]


class PaperFormatter:
    """
    Enforces strict paper structure and formatting
    """
    
    def __init__(self, required_sections=None):
        """
        Initialize Paper Formatter
        
        Args:
            required_sections: List of required sections (default: PAPER_SECTIONS)
        """
        self.required_sections = required_sections or PAPER_SECTIONS
    
    def validate_structure(self, sections: List[Dict[str, str]]) -> Dict:
        """
        Validate paper structure
        
        Args:
            sections: List of section dictionaries with 'title' and 'content'
        
        Returns:
            dict: Validation result with 'valid', 'errors', 'warnings'
        """
        errors = []
        warnings = []
        
        # Extract section titles
        section_titles = [s.get('title', '').strip() for s in sections]
        
        # Check for missing sections
        for required in self.required_sections:
            if required not in section_titles:
                errors.append(f"Missing required section: {required}")
        
        # Check section order
        present_required = [s for s in section_titles if s in self.required_sections]
        expected_order = [s for s in self.required_sections if s in present_required]
        
        if present_required != expected_order:
            warnings.append(f"Sections out of order. Expected: {expected_order}, Got: {present_required}")
        
        # Check for empty sections
        for section in sections:
            title = section.get('title', '')
            content = section.get('content', '').strip()
            
            if not content or len(content) < 50:
                warnings.append(f"Section '{title}' is too short or empty")
        
        # Check for duplicate sections
        title_counts = {}
        for title in section_titles:
            title_counts[title] = title_counts.get(title, 0) + 1
        
        for title, count in title_counts.items():
            if count > 1:
                errors.append(f"Duplicate section: {title} (appears {count} times)")
        
        is_valid = len(errors) == 0
        
        return {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'section_count': len(sections),
            'missing_sections': [s for s in self.required_sections if s not in section_titles]
        }
    
    def format_paper(self, sections: List[Dict[str, str]]) -> str:
        """
        Format sections into a complete paper
        
        Args:
            sections: List of section dictionaries
        
        Returns:
            str: Formatted paper text
        """
        paper = []
        
        for section in sections:
            title = section.get('title', 'Untitled')
            content = section.get('content', '')
            
            # Add section header
            paper.append(f"# {title}\n")
            paper.append(content.strip())
            paper.append("\n\n")
        
        return "".join(paper)
    
    def reorder_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Reorder sections to match required order
        
        Args:
            sections: List of section dictionaries
        
        Returns:
            list: Reordered sections
        """
        # Create mapping of title to section
        section_map = {s.get('title', ''): s for s in sections}
        
        # Reorder based on required sections
        reordered = []
        
        for required_title in self.required_sections:
            if required_title in section_map:
                reordered.append(section_map[required_title])
        
        # Add any extra sections at the end
        for section in sections:
            title = section.get('title', '')
            if title not in self.required_sections:
                reordered.append(section)
        
        return reordered
    
    def get_section_template(self, section_name: str) -> str:
        """
        Get template/guidelines for a specific section
        
        Args:
            section_name: Name of the section
        
        Returns:
            str: Template or guidelines
        """
        templates = {
            "Abstract": "Summarize the research problem, methodology, key results, and conclusions in 150-250 words.",
            "Introduction": "Introduce the problem, motivation, research questions, and paper structure.",
            "Related Work": "Review relevant prior work and position your contribution.",
            "Methodology": "Describe your approach, algorithms, datasets, and experimental setup in detail.",
            "Results": "Present experimental results with tables, figures, and statistical analysis.",
            "Discussion": "Interpret results, discuss limitations, and compare with related work.",
            "Conclusion": "Summarize contributions, findings, and future work.",
            "References": "List all cited works in proper academic format."
        }
        
        return templates.get(section_name, "No template available for this section.")


# Singleton instance
_paper_formatter = None


def get_paper_formatter():
    """
    Get singleton Paper Formatter instance
    
    Returns:
        PaperFormatter: Paper formatter instance
    """
    global _paper_formatter
    
    if _paper_formatter is None:
        _paper_formatter = PaperFormatter()
    
    return _paper_formatter


if __name__ == "__main__":
    # Test the formatter
    formatter = PaperFormatter()
    
    test_sections = [
        {"title": "Abstract", "content": "This paper presents a novel approach to deep learning."},
        {"title": "Introduction", "content": "Deep learning has revolutionized AI..."},
        {"title": "Methodology", "content": "We propose a new architecture..."},
        {"title": "Results", "content": "Our experiments show..."},
        {"title": "Conclusion", "content": "We have demonstrated..."}
    ]
    
    print("=" * 60)
    print("Testing Paper Formatter")
    print("=" * 60)
    
    # Validate structure
    validation = formatter.validate_structure(test_sections)
    
    print("\nValidation Results:")
    print(f"Valid: {validation['valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Missing sections: {validation['missing_sections']}")
    
    # Get template
    print("\nAbstract Template:")
    print(formatter.get_section_template("Abstract"))
