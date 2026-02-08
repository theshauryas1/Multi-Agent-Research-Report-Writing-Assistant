"""
Research Agent - Conducts web research and summarizes findings.
"""

from typing import Any, Dict, List

from .base_agent import BaseAgent
from utils.web_search import research_topic
from config import PROMPTS


class ResearchAgent(BaseAgent):
    """
    Agent responsible for conducting research on a given topic.
    
    This agent:
    1. Searches the web for relevant information
    2. Gathers information from multiple sources
    3. Summarizes findings into structured research notes
    """
    
    def __init__(self, verbose: bool = True):
        super().__init__(
            name="Research Agent",
            agent_type="research",
            verbose=verbose,
        )
    
    def run(self, topic: str, num_sources: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Conduct research on a topic.
        
        Args:
            topic: The topic to research
            num_sources: Number of sources to gather
            
        Returns:
            Dictionary containing:
                - research_summary: Summarized research findings
                - sources: List of source information
                - raw_data: Raw search results
        """
        self.log(f"Starting research on topic: '{topic}'")
        
        # Step 1: Search for information
        self.log("Searching for relevant sources...")
        search_data = research_topic(topic, num_sources)
        
        self.log(f"Found {search_data['source_count']} sources")
        
        # Step 2: Build context from search results
        context = self._build_research_context(search_data)
        
        # Step 3: Summarize with LLM
        self.log("Synthesizing research findings...")
        
        prompt = PROMPTS["research"].format(topic=topic)
        prompt += f"\n\nHere is the information gathered from various sources:\n\n{context}"
        
        try:
            research_summary = self.invoke_llm(prompt)
        except Exception as e:
            self.log(f"LLM summarization failed, using raw data: {e}", "warning")
            research_summary = self._fallback_summary(search_data)
        
        self.log("Research complete!")
        
        return {
            "research_summary": research_summary,
            "sources": search_data["sources"],
            "raw_data": search_data,
            "topic": topic,
        }
    
    def _build_research_context(self, search_data: Dict) -> str:
        """Build a context string from search results."""
        context_parts = []
        
        for i, source in enumerate(search_data["sources"], 1):
            context_parts.append(
                f"Source {i}: {source['title']}\n"
                f"URL: {source['url']}\n"
                f"Summary: {source['snippet']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _fallback_summary(self, search_data: Dict) -> str:
        """Create a fallback summary when LLM fails."""
        summary = f"# Research Summary: {search_data['topic']}\n\n"
        summary += "## Key Findings\n\n"
        
        for source in search_data["sources"]:
            summary += f"### {source['title']}\n"
            summary += f"{source['snippet']}\n\n"
        
        return summary


if __name__ == "__main__":
    # Test the research agent
    agent = ResearchAgent()
    result = agent.run("benefits of renewable energy")
    
    print("\n" + "="*50)
    print("RESEARCH RESULTS")
    print("="*50)
    print(f"\nSources found: {len(result['sources'])}")
    print("\nSummary preview:")
    print(result['research_summary'][:500] + "...")
