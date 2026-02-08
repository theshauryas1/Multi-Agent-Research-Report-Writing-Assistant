"""
Multi-Agent Research & Report Writing Assistant
Streamlit UI Application

A beautiful, interactive interface for generating research reports
using a multi-agent LangGraph pipeline.
"""

import streamlit as st
import os
import time
from datetime import datetime
from typing import Dict, Any

# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="Research Report Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import after page config
from config import MODEL_MODE, MODEL_CONFIGS, OUTPUT_DIR
from graph.workflow import run_workflow, create_workflow
from graph.state import create_initial_state
from utils.export import export_to_markdown, export_to_pdf, export_to_html


# ============== CUSTOM CSS ==============
def load_custom_css():
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-color: #00d4aa;
            --warning-color: #ffc107;
            --dark-bg: #1a1a2e;
        }
        
        /* Header styling */
        .main-header {
            background: var(--primary-gradient);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Agent status cards */
        .agent-card {
            background: linear-gradient(145deg, #2d2d44, #1a1a2e);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .agent-card:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
        }
        
        .agent-card.active {
            border-left-color: var(--success-color);
            animation: pulse 2s infinite;
        }
        
        .agent-card.complete {
            border-left-color: var(--success-color);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Log entries */
        .log-entry {
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
        }
        
        .log-timestamp {
            color: #888;
            margin-right: 10px;
        }
        
        .log-agent {
            color: #667eea;
            font-weight: 600;
            margin-right: 10px;
        }
        
        /* Report preview */
        .report-preview {
            background: white;
            color: #333;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 5px 30px rgba(0,0,0,0.1);
            max-height: 600px;
            overflow-y: auto;
        }
        
        /* Export buttons */
        .export-btn {
            background: var(--primary-gradient);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .export-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* Progress indicator */
        .progress-step {
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
        }
        
        .step-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-size: 14px;
        }
        
        .step-pending {
            background: #333;
            color: #666;
        }
        
        .step-active {
            background: var(--primary-gradient);
            color: white;
            animation: pulse 1s infinite;
        }
        
        .step-complete {
            background: var(--success-color);
            color: white;
        }
        
        /* Score display */
        .score-display {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)


# ============== SESSION STATE ==============
def init_session_state():
    """Initialize session state variables."""
    if "workflow_running" not in st.session_state:
        st.session_state.workflow_running = False
    if "current_state" not in st.session_state:
        st.session_state.current_state = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    if "final_report" not in st.session_state:
        st.session_state.final_report = ""


# ============== SIDEBAR ==============
def render_sidebar():
    """Render the sidebar with settings and status."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model mode display
        mode_labels = {
            "free": "🟢 Free (HuggingFace)",
            "local": "🟡 Local (Ollama)",
            "paid": "🔴 Paid (OpenAI)",
        }
        current_mode = mode_labels.get(MODEL_MODE, MODEL_MODE)
        st.info(f"**Model Mode:** {current_mode}")
        
        st.markdown("---")
        
        # API status
        st.markdown("### 🔐 API Status")
        
        from config import HUGGINGFACE_API_KEY, OPENAI_API_KEY, SERPAPI_API_KEY
        
        hf_status = "✅ Set" if HUGGINGFACE_API_KEY else "❌ Missing"
        openai_status = "✅ Set" if OPENAI_API_KEY else "❌ Missing"
        serp_status = "✅ Set" if SERPAPI_API_KEY else "⚠️ Using simulated search"
        
        st.markdown(f"- HuggingFace: {hf_status}")
        st.markdown(f"- OpenAI: {openai_status}")
        st.markdown(f"- SerpAPI: {serp_status}")
        
        st.markdown("---")
        
        # Workflow steps
        st.markdown("### 📋 Workflow Steps")
        
        steps = [
            ("🔍", "Research", "research"),
            ("📝", "Planning", "plan"),
            ("✍️", "Writing", "write"),
            ("🔎", "Review", "review"),
            ("🔧", "Fixing", "fix"),
            ("✅", "Finalize", "finalize"),
        ]
        
        current_status = st.session_state.get("current_state", {})
        current_step = current_status.get("status", "") if current_status else ""
        
        for icon, name, key in steps:
            if name.lower() in current_step.lower():
                st.markdown(f"🔵 **{icon} {name}** (Active)")
            elif st.session_state.report_generated:
                st.markdown(f"✅ {icon} {name}")
            else:
                st.markdown(f"⚪ {icon} {name}")
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **How to use:**
            1. Enter your topic in the main area
            2. Click "Generate Report"
            3. Watch the agents work in real-time
            4. Download your report when complete
            
            **Tips:**
            - Be specific with your topic
            - Complex topics may take longer
            - Check the logs for detailed progress
            """)


# ============== MAIN CONTENT ==============
def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>📚 Research Report Assistant</h1>
        <p>Powered by Multi-Agent LangGraph Pipeline</p>
    </div>
    """, unsafe_allow_html=True)


def render_topic_input():
    """Render the topic input section."""
    st.markdown("### 🎯 Enter Your Topic")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        topic = st.text_input(
            "Topic",
            placeholder="e.g., Benefits of Renewable Energy, History of Artificial Intelligence...",
            label_visibility="collapsed",
        )
    
    with col2:
        generate_btn = st.button(
            "🚀 Generate",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.workflow_running,
        )
    
    return topic, generate_btn


def render_progress():
    """Render the progress section during generation."""
    if st.session_state.workflow_running:
        with st.spinner("Generating your report..."):
            progress_placeholder = st.empty()
            logs_placeholder = st.container()
            
            return progress_placeholder, logs_placeholder
    return None, None


def render_logs():
    """Render the logs section."""
    if st.session_state.logs:
        with st.expander("📜 Agent Logs", expanded=True):
            log_container = st.container()
            with log_container:
                for log in st.session_state.logs[-20:]:  # Show last 20 logs
                    timestamp = log.get("timestamp", "")
                    agent = log.get("agent", "")
                    message = log.get("message", "")
                    
                    # Format based on agent type
                    if "Research" in agent:
                        icon = "🔍"
                    elif "Planner" in agent:
                        icon = "📝"
                    elif "Writer" in agent:
                        icon = "✍️"
                    elif "Reviewer" in agent:
                        icon = "🔎"
                    elif "Fixer" in agent:
                        icon = "🔧"
                    else:
                        icon = "📌"
                    
                    st.markdown(f"`{timestamp}` {icon} **{agent}**: {message}")


def render_report():
    """Render the generated report."""
    if st.session_state.report_generated and st.session_state.final_report:
        st.markdown("---")
        st.markdown("### 📄 Generated Report")
        
        # Score display if available
        current_state = st.session_state.current_state
        if current_state and "overall_score" in current_state:
            score = current_state["overall_score"]
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.metric("Quality Score", f"{score:.1f}/10")
        
        # Report tabs
        tab1, tab2 = st.tabs(["📖 Preview", "📝 Raw Markdown"])
        
        with tab1:
            st.markdown(st.session_state.final_report)
        
        with tab2:
            st.code(st.session_state.final_report, language="markdown")
        
        # Export section
        st.markdown("---")
        st.markdown("### 💾 Export Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Markdown", use_container_width=True):
                topic = current_state.get("topic", "report") if current_state else "report"
                filepath = export_to_markdown(
                    title=topic,
                    content=st.session_state.final_report,
                    sources=current_state.get("sources", []) if current_state else [],
                )
                st.success(f"Saved to: {filepath}")
                
                # Offer download
                with open(filepath, "r", encoding="utf-8") as f:
                    st.download_button(
                        "⬇️ Download",
                        f.read(),
                        file_name=os.path.basename(filepath),
                        mime="text/markdown",
                    )
        
        with col2:
            if st.button("📕 Download PDF", use_container_width=True):
                topic = current_state.get("topic", "report") if current_state else "report"
                filepath = export_to_pdf(
                    title=topic,
                    content=st.session_state.final_report,
                    sources=current_state.get("sources", []) if current_state else [],
                )
                st.success(f"Saved to: {filepath}")
                
                with open(filepath, "rb") as f:
                    st.download_button(
                        "⬇️ Download",
                        f.read(),
                        file_name=os.path.basename(filepath),
                        mime="application/pdf",
                    )
        
        with col3:
            if st.button("🌐 Download HTML", use_container_width=True):
                topic = current_state.get("topic", "report") if current_state else "report"
                filepath = export_to_html(
                    title=topic,
                    content=st.session_state.final_report,
                    sources=current_state.get("sources", []) if current_state else [],
                )
                st.success(f"Saved to: {filepath}")
                
                with open(filepath, "r", encoding="utf-8") as f:
                    st.download_button(
                        "⬇️ Download",
                        f.read(),
                        file_name=os.path.basename(filepath),
                        mime="text/html",
                    )


def run_generation(topic: str):
    """Run the report generation workflow."""
    st.session_state.workflow_running = True
    st.session_state.logs = []
    st.session_state.report_generated = False
    
    try:
        # Create workflow and initial state
        app = create_workflow()
        initial_state = create_initial_state(topic)
        
        # Create placeholders for live updates
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Run with streaming
        step_count = 0
        total_steps = 6  # research, plan, write, review, (fix), finalize
        
        for output in app.stream(initial_state):
            for node_name, state_update in output.items():
                step_count += 1
                progress = min(step_count / total_steps, 1.0)
                progress_bar.progress(progress)
                
                status = state_update.get("status", f"Running {node_name}")
                status_placeholder.info(f"🔄 {status}")
                
                # Update logs
                new_logs = state_update.get("logs", [])
                st.session_state.logs = new_logs
                
                # Update current state
                st.session_state.current_state = state_update
                
                # Check for final report
                if "final_report" in state_update and state_update["final_report"]:
                    st.session_state.final_report = state_update["final_report"]
        
        st.session_state.report_generated = True
        status_placeholder.success("✅ Report generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.exception(e)
    
    finally:
        st.session_state.workflow_running = False


# ============== MAIN APP ==============
def main():
    """Main application entry point."""
    load_custom_css()
    init_session_state()
    
    # Sidebar
    render_sidebar()
    
    # Main content
    render_header()
    
    # Topic input
    topic, generate_btn = render_topic_input()
    
    # Handle generation
    if generate_btn and topic:
        run_generation(topic)
    elif generate_btn and not topic:
        st.warning("Please enter a topic first!")
    
    # Show logs
    render_logs()
    
    # Show report
    render_report()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Built with ❤️ using LangGraph, LangChain, and Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
