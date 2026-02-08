"""
Configuration module for Multi-Agent Research & Report Writing Assistant.
Handles model mode switching and API key management.
"""

import os
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv()


class ModelMode(Enum):
    """Available model modes for the system."""
    FREE = "free"      # HuggingFace API (free tier)
    LOCAL = "local"    # Ollama (local models)
    PAID = "paid"      # OpenAI / Anthropic


# ============== MAIN CONFIGURATION ==============

# Set the model mode here: "free", "local", or "paid"
MODEL_MODE = os.getenv("MODEL_MODE", "free")

# Maximum revision iterations for the reviewer-fixer loop
MAX_REVISIONS = 3

# Minimum review score (1-10) to accept without revision
MIN_REVIEW_SCORE = 7

# ============== API KEYS ==============

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

# ============== MODEL MAPPINGS ==============

# Models available for each mode
MODEL_CONFIGS = {
    ModelMode.FREE.value: {
        "provider": "huggingface",
        "models": {
            "default": "mistralai/Mistral-7B-Instruct-v0.2",
            "research": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "writer": "HuggingFaceH4/zephyr-7b-beta",
            "reviewer": "openchat/openchat-3.5-0106",
        },
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    ModelMode.LOCAL.value: {
        "provider": "ollama",
        "models": {
            "default": "mistral",
            "research": "mixtral",
            "writer": "llama2",
            "reviewer": "gemma",
        },
        "temperature": 0.7,
        "max_tokens": 2048,
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    },
    ModelMode.PAID.value: {
        "provider": "openai",
        "models": {
            "default": "gpt-3.5-turbo",
            "research": "gpt-4",
            "writer": "gpt-4",
            "reviewer": "gpt-3.5-turbo",
        },
        "temperature": 0.7,
        "max_tokens": 4096,
    },
}

# ============== PROMPT TEMPLATES ==============

PROMPTS = {
    "research": """You are a research assistant. Given a topic, conduct thorough research and provide:
1. A comprehensive summary of key information
2. Important facts and statistics
3. Different perspectives on the topic
4. Relevant sources and references

Topic: {topic}

Provide well-structured research notes that can be used to write an informative report.""",

    "planner": """You are a report planner. Based on the research provided, create a detailed outline for a comprehensive report.

Topic: {topic}

Research Summary:
{research}

Create an outline with:
1. A compelling introduction section
2. 3-5 main body sections with clear themes
3. A conclusion section
4. For each section, provide a brief description of what it should cover

Format your response as a structured outline.""",

    "writer": """You are an expert writer. Write a detailed, engaging section for a report.

Topic: {topic}
Section Title: {section_title}
Section Description: {section_description}

Research Context:
{research}

Previous Sections (for context):
{previous_sections}

Write a well-structured, informative section that:
- Flows naturally from previous sections
- Uses clear, professional language
- Includes relevant facts and examples
- Is approximately 300-500 words""",

    "reviewer": """You are an expert editor and reviewer. Review the following report section.

Section Title: {section_title}
Content:
{content}

Evaluate the section on these criteria (score 1-10 for each):
1. Clarity: Is the writing clear and easy to understand?
2. Structure: Is the content well-organized?
3. Completeness: Does it cover the topic adequately?
4. Tone: Is the tone professional and appropriate?
5. Accuracy: Does the content seem factually accurate?

Provide:
- An overall score (average of all criteria)
- Specific feedback for improvement
- Whether this section needs revision (Yes/No)

Format as:
SCORE: [number]
NEEDS_REVISION: [Yes/No]
FEEDBACK: [detailed feedback]""",

    "fixer": """You are an expert editor. Revise the following section based on the feedback provided.

Original Section:
{original_content}

Review Feedback:
{feedback}

Rewrite the section to address all the feedback while maintaining the core information.
Provide only the revised section content, no explanations.""",
}

# ============== OUTPUT SETTINGS ==============

OUTPUT_DIR = "outputs"
DEFAULT_EXPORT_FORMAT = "markdown"  # "markdown", "pdf", or "html"


def get_current_config():
    """Get the configuration for the current model mode."""
    return MODEL_CONFIGS.get(MODEL_MODE, MODEL_CONFIGS[ModelMode.FREE.value])


def validate_config():
    """Validate that required API keys are set for the current mode."""
    config = get_current_config()
    provider = config["provider"]
    
    if provider == "openai" and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")
    
    if provider == "huggingface" and not HUGGINGFACE_API_KEY:
        raise ValueError("HuggingFace API key not set. Please set HUGGINGFACE_API_KEY environment variable.")
    
    return True
