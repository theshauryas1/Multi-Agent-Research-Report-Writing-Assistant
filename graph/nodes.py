"""
Graph Nodes - Individual node functions for the LangGraph workflow.
Each node wraps an agent and updates the shared state.
"""

from typing import Dict, Any
from datetime import datetime

from .state import ReportState, add_log
from agents import (
    ResearchAgent,
    PlannerAgent,
    WriterAgent,
    ReviewerAgent,
    FixerAgent,
    GroundingAgent,
)


def research_node(state: ReportState) -> Dict[str, Any]:
    """
    Research node - Conducts web research on the topic.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state fields
    """
    topic = state["topic"]
    
    # Create and run the research agent
    agent = ResearchAgent(verbose=True)
    result = agent.run(topic=topic)
    
    # Collect agent logs
    agent_logs = agent.get_logs()
    
    return {
        "research_summary": result["research_summary"],
        "sources": result["sources"],
        "status": "Research complete",
        "logs": state.get("logs", []) + agent_logs,
    }


def plan_node(state: ReportState) -> Dict[str, Any]:
    """
    Planning node - Creates the report outline.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state fields
    """
    topic = state["topic"]
    research = state["research_summary"]
    
    # Create and run the planner agent
    agent = PlannerAgent(verbose=True)
    result = agent.run(
        topic=topic,
        research_summary=research,
    )
    
    # Collect agent logs
    agent_logs = agent.get_logs()
    
    return {
        "outline": result["outline"],
        "outline_text": result["outline_text"],
        "status": "Outline created",
        "logs": state.get("logs", []) + agent_logs,
    }


def write_node(state: ReportState) -> Dict[str, Any]:
    """
    Writing node - Writes all sections of the report.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state fields
    """
    topic = state["topic"]
    outline = state["outline"]
    research = state["research_summary"]
    
    # Create and run the writer agent
    agent = WriterAgent(verbose=True)
    result = agent.run(
        topic=topic,
        outline=outline,
        research_summary=research,
    )
    
    # Collect agent logs
    agent_logs = agent.get_logs()
    
    return {
        "sections": result["sections"],
        "draft": result["full_draft"],
        "status": "Draft written",
        "logs": state.get("logs", []) + agent_logs,
    }


def review_node(state: ReportState) -> Dict[str, Any]:
    """
    Review node - Reviews all sections for quality.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state fields
    """
    sections = state["sections"]
    
    # Create and run the reviewer agent
    agent = ReviewerAgent(verbose=True)
    result = agent.run(sections=sections)
    
    # Collect agent logs
    agent_logs = agent.get_logs()
    
    # Increment iteration counter
    current_iteration = state.get("iteration", 0) + 1
    
    return {
        "reviews": result["reviews"],
        "overall_score": result["overall_score"],
        "needs_revision": result["needs_revision"],
        "sections_to_fix": result["sections_to_fix"],
        "iteration": current_iteration,
        "status": f"Review complete (iteration {current_iteration})",
        "logs": state.get("logs", []) + agent_logs,
    }


def fix_node(state: ReportState) -> Dict[str, Any]:
    """
    Fix node - Revises sections based on review feedback.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state fields
    """
    sections = state["sections"]
    reviews = state["reviews"]
    sections_to_fix = state["sections_to_fix"]
    
    # Create and run the fixer agent
    agent = FixerAgent(verbose=True)
    result = agent.run(
        sections=sections,
        reviews=reviews,
        sections_to_fix=sections_to_fix,
    )
    
    # Rebuild the draft with fixed sections
    topic = state["topic"]
    fixed_draft = agent.compile_final_draft(topic, result["fixed_sections"])
    
    # Collect agent logs
    agent_logs = agent.get_logs()
    
    return {
        "sections": result["fixed_sections"],
        "draft": fixed_draft,
        "status": "Revisions complete",
        "logs": state.get("logs", []) + agent_logs,
    }


def finalize_node(state: ReportState) -> Dict[str, Any]:
    """
    Finalize node - Prepares the final report.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state fields
    """
    draft = state["draft"]
    sources = state.get("sources", [])
    
    # Add sources section to the final report
    final_report = draft
    
    if sources:
        final_report += "\n\n---\n\n## References\n\n"
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            if url:
                final_report += f"{i}. [{title}]({url})\n"
            else:
                final_report += f"{i}. {title}\n"
    
    log_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "agent": "Finalizer",
        "message": "Report finalized successfully!",
    }
    
    return {
        "final_report": final_report,
        "status": "Complete",
        "logs": state.get("logs", []) + [log_entry],
    }


def grounding_node(state: ReportState) -> Dict[str, Any]:
    """
    Grounding node - Validates claims against retrieved sources.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state fields with grounding metrics
    """
    sections = state["sections"]
    sources = state.get("sources", [])
    
    # Create and run the grounding agent
    agent = GroundingAgent(verbose=True)
    result = agent.run(
        sections=sections,
        sources=sources,
        retrieval_context=state.get("retrieval_context", {}),
    )
    
    # Collect agent logs
    agent_logs = agent.get_logs()
    
    # Update metrics
    current_metrics = state.get("metrics", {})
    current_metrics.update({
        "grounding_accuracy": result["grounding_score"],
        "hallucination_rate": result["hallucination_rate"],
        "total_claims": result.get("total_claims", 0),
        "grounded_claims": result.get("grounded_claims", 0),
    })
    
    return {
        "grounding_score": result["grounding_score"],
        "hallucination_rate": result["hallucination_rate"],
        "metrics": current_metrics,
        "status": f"Grounding validated ({result['grounding_score']:.0%})",
        "logs": state.get("logs", []) + agent_logs,
    }


def should_continue_review(state: ReportState) -> str:
    """
    Conditional edge function - Determines if more revisions are needed.
    
    Args:
        state: Current workflow state
        
    Returns:
        "fix" if revisions needed and under max iterations,
        "finalize" otherwise
    """
    from config import MAX_REVISIONS
    
    needs_revision = state.get("needs_revision", False)
    iteration = state.get("iteration", 0)
    
    if needs_revision and iteration < MAX_REVISIONS:
        return "fix"
    else:
        return "finalize"

