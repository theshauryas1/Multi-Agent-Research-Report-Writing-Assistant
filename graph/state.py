"""
State Schema for LangGraph Workflow.
Defines the shared state that flows through all agent nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field


class ReportState(TypedDict, total=False):
    """
    Shared state for the report generation workflow.
    
    This state is passed through all nodes in the LangGraph
    and accumulates information as the report is generated.
    """
    
    # Input
    topic: str
    """The user's input topic for the report."""
    
    # Research phase outputs
    research_summary: str
    """Summarized research findings from the Research Agent."""
    
    sources: List[Dict[str, str]]
    """List of source information with 'title', 'url', 'snippet'."""
    
    # Planning phase outputs
    outline: List[Dict[str, str]]
    """Structured outline with section 'title' and 'description'."""
    
    outline_text: str
    """Text representation of the outline."""
    
    # Writing phase outputs
    sections: List[Dict[str, Any]]
    """List of written sections with 'title', 'content', etc."""
    
    draft: str
    """Complete draft of the report."""
    
    # Review phase outputs
    reviews: List[Dict[str, Any]]
    """Review feedback for each section."""
    
    overall_score: float
    """Average score across all sections (1-10)."""
    
    needs_revision: bool
    """Whether any sections need revision."""
    
    sections_to_fix: List[int]
    """Indices of sections that need fixing."""
    
    # Grounding phase outputs
    grounding_score: float
    """Overall claim-to-source grounding accuracy (0-1)."""
    
    hallucination_rate: float
    """Rate of hallucinated citations (0-1)."""
    
    retrieval_context: Dict[str, List[Dict[str, Any]]]
    """Per-section retrieved chunks for grounding."""
    
    # Metrics
    metrics: Dict[str, Any]
    """Accumulated evaluation metrics for logging."""
    
    # Final output
    final_report: str
    """The finalized report content."""
    
    # Workflow control
    iteration: int
    """Current revision iteration count."""
    
    status: str
    """Current status message for UI display."""
    
    error: Optional[str]
    """Error message if something went wrong."""
    
    logs: List[Dict[str, str]]
    """Accumulated logs from all agents."""


def create_initial_state(topic: str) -> ReportState:
    """
    Create an initial state for the workflow.
    
    Args:
        topic: The topic to generate a report on
        
    Returns:
        Initialized ReportState
    """
    return ReportState(
        topic=topic,
        research_summary="",
        sources=[],
        outline=[],
        outline_text="",
        sections=[],
        draft="",
        reviews=[],
        overall_score=0.0,
        needs_revision=False,
        sections_to_fix=[],
        grounding_score=0.0,
        hallucination_rate=0.0,
        retrieval_context={},
        metrics={},
        final_report="",
        iteration=0,
        status="initialized",
        error=None,
        logs=[],
    )


def add_log(state: ReportState, agent: str, message: str) -> ReportState:
    """
    Add a log entry to the state.
    
    Args:
        state: Current state
        agent: Name of the agent logging
        message: Log message
        
    Returns:
        Updated state with new log entry
    """
    from datetime import datetime
    
    log_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "message": message,
    }
    
    # Create new logs list with the entry
    new_logs = state.get("logs", []) + [log_entry]
    
    return {**state, "logs": new_logs}
