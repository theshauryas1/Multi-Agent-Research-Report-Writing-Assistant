"""
Metrics Module - Evaluation and logging for research report generation.

Provides automatic tracking of:
- Claim-to-source grounding accuracy
- Hallucination rate
- Structural coherence
- Redundancy metrics
"""

from .logger import MetricsLogger, get_logger
from .grounding import GroundingEvaluator, calculate_grounding_accuracy
from .hallucination import HallucinationDetector, calculate_hallucination_rate
from .coherence import CoherenceScorer, calculate_coherence_score

__all__ = [
    "MetricsLogger",
    "get_logger",
    "GroundingEvaluator",
    "calculate_grounding_accuracy",
    "HallucinationDetector", 
    "calculate_hallucination_rate",
    "CoherenceScorer",
    "calculate_coherence_score",
]
