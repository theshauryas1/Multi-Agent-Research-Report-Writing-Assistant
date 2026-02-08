"""
Workflow - Complete LangGraph StateGraph definition.
Orchestrates the multi-agent report generation pipeline.
"""

from typing import Dict, Any, Optional, Generator
from langgraph.graph import StateGraph, END

from .state import ReportState, create_initial_state
from .nodes import (
    research_node,
    plan_node,
    write_node,
    review_node,
    fix_node,
    finalize_node,
    should_continue_review,
)


def create_workflow() -> StateGraph:
    """
    Create the complete report generation workflow.
    
    The workflow follows this pattern:
    1. Research: Gather information on the topic
    2. Plan: Create an outline
    3. Write: Write all sections
    4. Review: Evaluate quality
    5. Fix (conditional): Revise if needed, then re-review
    6. Finalize: Prepare final output
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph with our state type
    workflow = StateGraph(ReportState)
    
    # Add all nodes
    workflow.add_node("research", research_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("write", write_node)
    workflow.add_node("review", review_node)
    workflow.add_node("fix", fix_node)
    workflow.add_node("finalize", finalize_node)
    
    # Define the flow
    # Start with research
    workflow.set_entry_point("research")
    
    # Linear flow: research -> plan -> write -> review
    workflow.add_edge("research", "plan")
    workflow.add_edge("plan", "write")
    workflow.add_edge("write", "review")
    
    # Conditional edge after review
    workflow.add_conditional_edges(
        "review",
        should_continue_review,
        {
            "fix": "fix",
            "finalize": "finalize",
        }
    )
    
    # After fixing, go back to review
    workflow.add_edge("fix", "review")
    
    # Finalize ends the workflow
    workflow.add_edge("finalize", END)
    
    # Compile and return
    return workflow.compile()


def run_workflow(
    topic: str,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete workflow for a given topic.
    
    Args:
        topic: The topic to generate a report on
        stream: If True, yields state updates as they happen
        
    Returns:
        Final state dictionary with the complete report
    """
    # Create initial state
    initial_state = create_initial_state(topic)
    
    # Create and run the workflow
    app = create_workflow()
    
    if stream:
        # Return a generator for streaming updates
        return _stream_workflow(app, initial_state)
    else:
        # Run to completion
        final_state = app.invoke(initial_state)
        return final_state


def _stream_workflow(
    app,
    initial_state: ReportState,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream workflow execution, yielding state updates.
    
    Args:
        app: Compiled workflow
        initial_state: Starting state
        
    Yields:
        State updates as they happen
    """
    for output in app.stream(initial_state):
        # Each output is a dict with node name as key
        for node_name, state_update in output.items():
            yield {
                "node": node_name,
                "update": state_update,
            }


def run_workflow_with_callback(
    topic: str,
    callback: callable,
) -> Dict[str, Any]:
    """
    Run workflow with a callback for status updates.
    
    Args:
        topic: The topic to generate a report on
        callback: Function to call with (node_name, status) updates
        
    Returns:
        Final state dictionary
    """
    # Create initial state
    initial_state = create_initial_state(topic)
    
    # Create the workflow
    app = create_workflow()
    
    # Stream and call callback
    final_state = None
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            status = state_update.get("status", f"Running {node_name}")
            callback(node_name, status)
            final_state = state_update
    
    return final_state


# Visualization helper (for debugging)
def get_workflow_diagram() -> str:
    """
    Get a text representation of the workflow.
    
    Returns:
        ASCII diagram of the workflow
    """
    return """
    ┌──────────────┐
    │   START      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Research    │
    │    Agent     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Planner    │
    │    Agent     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   Writer     │
    │    Agent     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Reviewer    │◄─────────┐
    │    Agent     │          │
    └──────┬───────┘          │
           │                  │
           ▼                  │
    ┌──────────────┐          │
    │ Score >= 7?  │──No──────┤
    └──────┬───────┘          │
           │                  │
          Yes                 │
           │           ┌──────┴───────┐
           │           │    Fixer     │
           │           │    Agent     │
           │           └──────────────┘
           ▼
    ┌──────────────┐
    │   Finalize   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │     END      │
    └──────────────┘
    """


if __name__ == "__main__":
    # Test the workflow
    print("Testing workflow creation...")
    app = create_workflow()
    print("✓ Workflow created successfully!")
    
    print("\nWorkflow diagram:")
    print(get_workflow_diagram())
    
    print("\nTo run the workflow:")
    print("  from graph.workflow import run_workflow")
    print("  result = run_workflow('Your Topic Here')")
