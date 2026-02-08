"""
LLM Factory - Dynamic model provider based on configuration.
Supports HuggingFace (free), Ollama (local), and OpenAI (paid).
"""

from typing import Optional
from langchain_core.language_models.base import BaseLanguageModel

import sys
sys.path.append('..')

from config import (
    MODEL_MODE,
    MODEL_CONFIGS,
    OPENAI_API_KEY,
    HUGGINGFACE_API_KEY,
    get_current_config,
)


def get_llm(
    agent_type: str = "default",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseLanguageModel:
    """
    Get an LLM instance based on the current MODEL_MODE configuration.
    
    Args:
        agent_type: Type of agent requesting the LLM 
                   ("research", "writer", "reviewer", or "default")
        temperature: Override default temperature if provided
        max_tokens: Override default max_tokens if provided
    
    Returns:
        A LangChain LLM instance configured for the current mode
    
    Raises:
        ValueError: If the model mode is not supported or API keys are missing
    """
    config = get_current_config()
    provider = config["provider"]
    
    # Get model name for the agent type
    model_name = config["models"].get(agent_type, config["models"]["default"])
    
    # Use config defaults or overrides
    temp = temperature if temperature is not None else config["temperature"]
    tokens = max_tokens if max_tokens is not None else config["max_tokens"]
    
    if provider == "huggingface":
        return _get_huggingface_llm(model_name, temp, tokens)
    elif provider == "ollama":
        return _get_ollama_llm(model_name, temp, tokens, config.get("base_url"))
    elif provider == "openai":
        return _get_openai_llm(model_name, temp, tokens)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _get_huggingface_llm(model_name: str, temperature: float, max_tokens: int):
    """Create a HuggingFace Hub LLM instance."""
    try:
        from langchain_huggingface import HuggingFaceEndpoint
    except ImportError:
        from langchain_community.llms import HuggingFaceHub as HuggingFaceEndpoint
    
    if not HUGGINGFACE_API_KEY:
        raise ValueError(
            "HuggingFace API key not found. "
            "Please set HUGGINGFACE_API_KEY in your environment or .env file."
        )
    
    return HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=temperature,
        max_new_tokens=max_tokens,
    )


def _get_ollama_llm(model_name: str, temperature: float, max_tokens: int, base_url: str):
    """Create an Ollama LLM instance."""
    from langchain_community.llms import Ollama
    
    return Ollama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        num_predict=max_tokens,
    )


def _get_openai_llm(model_name: str, temperature: float, max_tokens: int):
    """Create an OpenAI LLM instance."""
    from langchain_openai import ChatOpenAI
    
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found. "
            "Please set OPENAI_API_KEY in your environment or .env file."
        )
    
    return ChatOpenAI(
        model=model_name,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_available_models() -> dict:
    """Get a dictionary of available models for the current mode."""
    config = get_current_config()
    return {
        "mode": MODEL_MODE,
        "provider": config["provider"],
        "models": config["models"],
    }


def test_connection() -> bool:
    """
    Test if the current LLM configuration is working.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        llm = get_llm()
        response = llm.invoke("Say 'Connection successful' in exactly those words.")
        return "successful" in response.lower() if isinstance(response, str) else True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the LLM factory
    print(f"Current mode: {MODEL_MODE}")
    print(f"Available models: {get_available_models()}")
    print(f"Testing connection...")
    if test_connection():
        print("✓ Connection successful!")
    else:
        print("✗ Connection failed!")
