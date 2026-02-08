"""
Grounding Agent - Validates claims against retrieved sources.
Ensures all factual statements are supported by retrieved documents.
"""

from typing import Any, Dict, List

from .base_agent import BaseAgent
from config import PROMPTS


class GroundingAgent(BaseAgent):
    """
    Agent responsible for validating claim-to-source grounding.
    
    This agent:
    1. Extracts factual claims from generated sections
    2. Matches claims against retrieved source chunks
    3. Flags unsupported statements
    4. Returns grounding score and details
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="Grounding Agent",
            agent_type="reviewer",  # Use reviewer model for evaluation
            verbose=verbose,
        )
        self._grounding_evaluator = None
        self._hallucination_detector = None
    
    @property
    def grounding_evaluator(self):
        """Lazy load grounding evaluator."""
        if self._grounding_evaluator is None:
            from metrics.grounding import GroundingEvaluator
            self._grounding_evaluator = GroundingEvaluator()
        return self._grounding_evaluator
    
    @property
    def hallucination_detector(self):
        """Lazy load hallucination detector."""
        if self._hallucination_detector is None:
            from metrics.hallucination import HallucinationDetector
            self._hallucination_detector = HallucinationDetector()
        return self._hallucination_detector
    
    def run(
        self,
        sections: List[Dict[str, str]],
        sources: List[Dict[str, Any]],
        retrieval_context: Dict[str, List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate grounding of all sections against sources.
        
        Args:
            sections: List of section dictionaries with 'title' and 'content'
            sources: List of source dictionaries from research
            retrieval_context: Optional per-section retrieved chunks
            
        Returns:
            Dictionary containing:
                - grounding_score: Overall grounding accuracy (0-1)
                - hallucination_rate: Rate of fabricated citations
                - section_results: Per-section grounding details
                - ungrounded_claims: List of unsupported claims
                - needs_revision: Whether grounding issues require fixes
        """
        self.log("Starting grounding validation...")
        
        section_results = []
        all_ungrounded_claims = []
        all_hallucinated_citations = []
        
        for i, section in enumerate(sections):
            self.log(f"Validating section {i+1}/{len(sections)}: {section['title']}")
            
            # Get section-specific sources if available
            section_sources = sources
            if retrieval_context and section['title'] in retrieval_context:
                section_sources = retrieval_context[section['title']]
            
            # Evaluate grounding
            grounding_result = self.grounding_evaluator.evaluate(
                generated_text=section['content'],
                sources=section_sources,
                section_name=section['title'],
            )
            
            # Check for hallucinated citations
            hallucination_result = self.hallucination_detector.detect(
                generated_text=section['content'],
                sources=section_sources,
            )
            
            # Collect ungrounded claims
            ungrounded = [
                c for c in grounding_result.get('claims', [])
                if not c.get('is_grounded', True)
            ]
            all_ungrounded_claims.extend(ungrounded)
            
            # Collect hallucinated citations
            hallucinated = [
                c for c in hallucination_result.get('citations', [])
                if not c.get('is_valid', True)
            ]
            all_hallucinated_citations.extend(hallucinated)
            
            section_results.append({
                'title': section['title'],
                'grounding_accuracy': grounding_result['accuracy'],
                'hallucination_rate': hallucination_result['hallucination_rate'],
                'total_claims': grounding_result['total_claims'],
                'grounded_claims': grounding_result['grounded_claims'],
                'ungrounded_claims': len(ungrounded),
                'hallucinated_citations': len(hallucinated),
            })
            
            accuracy = grounding_result['accuracy']
            if accuracy >= 0.9:
                self.log(f"  → Section well-grounded ({accuracy:.0%})")
            elif accuracy >= 0.7:
                self.log(f"  → Section partially grounded ({accuracy:.0%})", "warning")
            else:
                self.log(f"  → Section poorly grounded ({accuracy:.0%})", "error")
        
        # Calculate overall metrics
        total_claims = sum(r['total_claims'] for r in section_results)
        grounded_claims = sum(r['grounded_claims'] for r in section_results)
        
        overall_grounding = grounded_claims / total_claims if total_claims > 0 else 1.0
        
        total_citations_checked = sum(
            r.get('total_claims', 0) for r in section_results
        )
        total_hallucinated = len(all_hallucinated_citations)
        overall_hallucination = total_hallucinated / max(total_citations_checked, 1)
        
        # Determine if revision needed
        needs_revision = (
            overall_grounding < 0.85 or  # Below 85% grounding
            len(all_hallucinated_citations) > 0  # Any hallucinated citations
        )
        
        self.log(f"Grounding validation complete")
        self.log(f"  Overall grounding: {overall_grounding:.1%}")
        self.log(f"  Hallucination rate: {overall_hallucination:.1%}")
        
        if needs_revision:
            self.log(f"  ⚠ {len(all_ungrounded_claims)} ungrounded claims need review")
        
        return {
            'grounding_score': overall_grounding,
            'hallucination_rate': overall_hallucination,
            'section_results': section_results,
            'ungrounded_claims': all_ungrounded_claims,
            'hallucinated_citations': all_hallucinated_citations,
            'needs_revision': needs_revision,
            'total_claims': total_claims,
            'grounded_claims': grounded_claims,
        }
    
    def validate_section(
        self,
        section_title: str,
        content: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate a single section's grounding.
        
        Args:
            section_title: Title of the section
            content: Section content to validate
            sources: Source chunks to validate against
            
        Returns:
            Grounding result for this section
        """
        grounding_result = self.grounding_evaluator.evaluate(
            generated_text=content,
            sources=sources,
            section_name=section_title,
        )
        
        hallucination_result = self.hallucination_detector.detect(
            generated_text=content,
            sources=sources,
        )
        
        return {
            'title': section_title,
            'grounding_accuracy': grounding_result['accuracy'],
            'hallucination_rate': hallucination_result['hallucination_rate'],
            'claims': grounding_result.get('claims', []),
            'citations': hallucination_result.get('citations', []),
        }
    
    def get_revision_suggestions(
        self,
        ungrounded_claims: List[Dict],
        sources: List[Dict],
    ) -> str:
        """
        Generate suggestions for fixing ungrounded claims.
        
        Uses LLM to suggest how to revise or remove ungrounded statements.
        """
        if not ungrounded_claims:
            return "No ungrounded claims to revise."
        
        claims_text = "\n".join([
            f"- {c.get('text', 'Unknown claim')}"
            for c in ungrounded_claims[:5]  # Limit to top 5
        ])
        
        sources_text = "\n".join([
            f"- {s.get('title', s.get('content', '')[:100])}"
            for s in sources[:5]
        ])
        
        prompt = f"""The following claims in a generated report are not supported by the available sources:

Ungrounded Claims:
{claims_text}

Available Sources:
{sources_text}

For each ungrounded claim, provide a brief suggestion:
1. Should it be removed?
2. Can it be rephrased to be supported by sources?
3. Should it be marked as needing a citation?

Provide concise suggestions."""

        try:
            suggestions = self.invoke_llm(prompt)
            return suggestions
        except Exception as e:
            self.log(f"Failed to generate suggestions: {e}", "warning")
            return "Could not generate revision suggestions."


if __name__ == "__main__":
    # Test the grounding agent
    agent = GroundingAgent()
    
    test_sections = [
        {
            "title": "Introduction",
            "content": """
            Renewable energy accounts for approximately 30% of global electricity generation.
            Solar power capacity has grown by 50% over the last decade. This transition is
            essential for addressing climate change and reducing carbon emissions.
            """,
        },
        {
            "title": "Solar Energy",
            "content": """
            Photovoltaic cells convert sunlight directly into electricity with efficiency
            rates now exceeding 22% for commercial panels. The cost of solar installations
            has dropped by 80% since 2010, according to industry reports.
            """,
        },
    ]
    
    test_sources = [
        {
            "title": "Global Renewable Energy Report",
            "content": "Renewable energy now provides about 30% of global electricity.",
            "url": "https://example.com/renewable-report",
        },
        {
            "title": "Solar Technology Advances",
            "content": "Modern solar panels achieve efficiency rates of 20-22%.",
            "url": "https://example.com/solar-tech",
        },
    ]
    
    result = agent.run(sections=test_sections, sources=test_sources)
    
    print("\n" + "="*50)
    print("GROUNDING VALIDATION RESULTS")
    print("="*50)
    print(f"\nOverall grounding: {result['grounding_score']:.1%}")
    print(f"Hallucination rate: {result['hallucination_rate']:.1%}")
    print(f"Needs revision: {result['needs_revision']}")
    
    for section in result['section_results']:
        print(f"\n{section['title']}:")
        print(f"  Grounding: {section['grounding_accuracy']:.1%}")
        print(f"  Claims: {section['grounded_claims']}/{section['total_claims']} grounded")
