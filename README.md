# Multi-Agent Research & Report Writing Assistant

A LangGraph-powered multi-agent system that automates research, planning, writing, and reviewing of structured reports on any user-defined topic.

## 🎯 Features

- **Multi-Agent Pipeline**: Research → Plan → Write → Review → Fix → Finalize
- **3 Model Modes**: 
  - 🟢 **Free**: HuggingFace API (Mistral, Mixtral, Zephyr)
  - 🟡 **Local**: Ollama (Mistral, Llama2, Gemma)
  - 🔴 **Paid**: OpenAI (GPT-4, GPT-3.5-turbo)
- **LangGraph Orchestration**: State-based workflow with conditional revision loops
- **Beautiful Streamlit UI**: Real-time progress, live logs, and report preview
- **Multi-Format Export**: Markdown, PDF, and HTML

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
MODEL_MODE=free
HUGGINGFACE_API_KEY=your_key_here
```

### 3. Run the Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                    # Streamlit main application
├── config.py                 # Configuration and settings
├── requirements.txt          # Dependencies
│
├── agents/
│   ├── base_agent.py         # Base agent class
│   ├── research_agent.py     # Web research
│   ├── planner_agent.py      # Report outline
│   ├── writer_agent.py       # Section writing
│   ├── reviewer_agent.py     # Quality review
│   └── fixer_agent.py        # Revisions
│
├── graph/
│   ├── state.py              # LangGraph state schema
│   ├── nodes.py              # Graph node functions
│   └── workflow.py           # Complete workflow
│
├── utils/
│   ├── llm_factory.py        # LLM provider switching
│   ├── web_search.py         # Search functionality
│   └── export.py             # Export utilities
│
└── outputs/                  # Generated reports
```

## 🔧 Configuration

Edit `config.py` to customize:

- `MODEL_MODE`: `"free"`, `"local"`, or `"paid"`
- `MAX_REVISIONS`: Maximum revision iterations (default: 3)
- `MIN_REVIEW_SCORE`: Minimum score to pass review (default: 7)

## 🤖 Agent Overview

| Agent | Purpose |
|-------|---------|
| 🔍 Research | Searches web and summarizes findings |
| 📝 Planner | Creates logical report outline |
| ✍️ Writer | Writes each section |
| 🔎 Reviewer | Evaluates quality (score 1-10) |
| 🔧 Fixer | Revises sections based on feedback |

## 📊 Workflow Diagram

```
START → Research → Plan → Write → Review
                                    ↓
                           Score >= 7? ─No→ Fix ─┐
                                    ↓            │
                                   Yes          ↑
                                    ↓            │
                              Finalize ←────────┘
                                    ↓
                                  END
```

## 🛠️ Advanced Usage

### Programmatic API

```python
from graph.workflow import run_workflow

# Run complete workflow
result = run_workflow("Benefits of Renewable Energy")

# Access the final report
print(result["final_report"])

# Access sources
print(result["sources"])
```

### Streaming Updates

```python
from graph.workflow import run_workflow

# Stream updates
for update in run_workflow("Your Topic", stream=True):
    print(f"Node: {update['node']}")
    print(f"Status: {update['update'].get('status', '')}")
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for LLM integrations
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Streamlit](https://streamlit.io/) for the UI framework
