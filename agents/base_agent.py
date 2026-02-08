"""
Base Agent - Abstract base class for all agents in the system.
Provides common interface and utilities for agent implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from datetime import datetime

import sys
sys.path.append('..')

from utils.llm_factory import get_llm


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Each agent has a specific role and implements the run() method
    to process inputs and produce outputs.
    """
    
    def __init__(
        self,
        name: str,
        agent_type: str = "default",
        temperature: Optional[float] = None,
        verbose: bool = True,
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Human-readable name for the agent
            agent_type: Type for LLM selection (research, writer, reviewer, default)
            temperature: Override default temperature
            verbose: Whether to log agent activities
        """
        self.name = name
        self.agent_type = agent_type
        self.temperature = temperature
        self.verbose = verbose
        self.llm = None
        self._logs = []
        
        # Set up logging
        self.logger = logging.getLogger(f"Agent.{name}")
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    def _initialize_llm(self):
        """Initialize the LLM for this agent."""
        if self.llm is None:
            self.llm = get_llm(
                agent_type=self.agent_type,
                temperature=self.temperature,
            )
    
    def log(self, message: str, level: str = "info"):
        """
        Log a message from this agent.
        
        Args:
            message: The message to log
            level: Log level (info, warning, error)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "agent": self.name,
            "level": level,
            "message": message,
        }
        self._logs.append(log_entry)
        
        if self.verbose:
            log_func = getattr(self.logger, level, self.logger.info)
            log_func(f"[{timestamp}] {message}")
    
    def get_logs(self) -> list:
        """Get all logs from this agent."""
        return self._logs.copy()
    
    def clear_logs(self):
        """Clear all logs."""
        self._logs = []
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        
        Args:
            **kwargs: Agent-specific input parameters
            
        Returns:
            Dictionary containing the agent's output
        """
        pass
    
    def invoke_llm(self, prompt: str) -> str:
        """
        Invoke the LLM with a prompt and return the response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
        self._initialize_llm()
        
        try:
            response = self.llm.invoke(prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            self.log(f"LLM invocation failed: {e}", "error")
            raise
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.agent_type}')"
