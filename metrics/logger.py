"""
Metrics Logger - Persistent logging of evaluation metrics.
Tracks all metrics per run and enables comparison across runs.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class RunMetrics:
    """Metrics for a single report generation run."""
    
    # Identification
    run_id: str = ""
    timestamp: str = ""
    topic: str = ""
    
    # Generation metrics
    generation_time_seconds: float = 0.0
    total_sections: int = 0
    total_words: int = 0
    total_pages: float = 0.0  # Estimated at 500 words/page
    
    # Grounding metrics
    grounding_accuracy: float = 0.0
    total_claims: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    
    # Hallucination metrics
    hallucination_rate: float = 0.0
    total_citations: int = 0
    valid_citations: int = 0
    hallucinated_citations: int = 0
    
    # Coherence metrics
    structural_coherence: float = 0.0
    section_scores: List[float] = field(default_factory=list)
    
    # Redundancy metrics
    redundancy_rate: float = 0.0
    avg_section_similarity: float = 0.0
    
    # Revision metrics
    revision_iterations: int = 0
    initial_review_score: float = 0.0
    final_review_score: float = 0.0
    
    # Additional metadata
    model_used: str = ""
    retrieval_sources: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsLogger:
    """
    Logs and persists metrics for report generation runs.
    
    Features:
    - Automatic run ID generation
    - JSON persistence
    - Historical comparison
    """
    
    def __init__(self, output_dir: str = "outputs/metrics"):
        """
        Initialize the metrics logger.
        
        Args:
            output_dir: Directory to store metrics files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_run: Optional[RunMetrics] = None
        self._start_time: Optional[datetime] = None
    
    def start_run(self, topic: str, model: str = "") -> str:
        """
        Start a new metrics run.
        
        Args:
            topic: Report topic
            model: Model name used
            
        Returns:
            Run ID
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        self.current_run = RunMetrics(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            topic=topic,
            model_used=model,
        )
        
        self._start_time = datetime.now()
        
        print(f"📊 Started metrics logging: {run_id}")
        return run_id
    
    def end_run(self) -> RunMetrics:
        """
        End the current run and calculate final metrics.
        
        Returns:
            Finalized RunMetrics
        """
        if self.current_run is None:
            raise ValueError("No run in progress")
        
        # Calculate generation time
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            self.current_run.generation_time_seconds = elapsed
        
        # Save to file
        self._save_run()
        
        run = self.current_run
        self.current_run = None
        self._start_time = None
        
        print(f"📊 Completed metrics logging: {run.run_id}")
        return run
    
    def log_generation(
        self,
        sections: int,
        words: int,
    ) -> None:
        """Log generation statistics."""
        if self.current_run:
            self.current_run.total_sections = sections
            self.current_run.total_words = words
            self.current_run.total_pages = words / 500.0
    
    def log_grounding(
        self,
        accuracy: float,
        total_claims: int,
        grounded_claims: int,
    ) -> None:
        """Log grounding metrics."""
        if self.current_run:
            self.current_run.grounding_accuracy = accuracy
            self.current_run.total_claims = total_claims
            self.current_run.grounded_claims = grounded_claims
            self.current_run.ungrounded_claims = total_claims - grounded_claims
    
    def log_hallucination(
        self,
        rate: float,
        total_citations: int,
        valid_citations: int,
    ) -> None:
        """Log hallucination metrics."""
        if self.current_run:
            self.current_run.hallucination_rate = rate
            self.current_run.total_citations = total_citations
            self.current_run.valid_citations = valid_citations
            self.current_run.hallucinated_citations = total_citations - valid_citations
    
    def log_coherence(
        self,
        score: float,
        section_scores: List[float],
    ) -> None:
        """Log coherence metrics."""
        if self.current_run:
            self.current_run.structural_coherence = score
            self.current_run.section_scores = section_scores
    
    def log_redundancy(
        self,
        rate: float,
        avg_similarity: float,
    ) -> None:
        """Log redundancy metrics."""
        if self.current_run:
            self.current_run.redundancy_rate = rate
            self.current_run.avg_section_similarity = avg_similarity
    
    def log_revision(
        self,
        iterations: int,
        initial_score: float,
        final_score: float,
    ) -> None:
        """Log revision loop metrics."""
        if self.current_run:
            self.current_run.revision_iterations = iterations
            self.current_run.initial_review_score = initial_score
            self.current_run.final_review_score = final_score
    
    def log_retrieval(self, num_sources: int) -> None:
        """Log retrieval statistics."""
        if self.current_run:
            self.current_run.retrieval_sources = num_sources
    
    def _save_run(self) -> None:
        """Save current run to file."""
        if self.current_run is None:
            return
        
        filepath = self.output_dir / f"{self.current_run.run_id}.json"
        
        with open(filepath, "w") as f:
            json.dump(self.current_run.to_dict(), f, indent=2)
        
        print(f"💾 Saved metrics to {filepath}")
    
    def get_run(self, run_id: str) -> Optional[RunMetrics]:
        """Load a specific run's metrics."""
        filepath = self.output_dir / f"{run_id}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return RunMetrics(**data)
    
    def get_all_runs(self) -> List[RunMetrics]:
        """Load all saved runs."""
        runs = []
        
        for filepath in self.output_dir.glob("run_*.json"):
            with open(filepath, "r") as f:
                data = json.load(f)
            runs.append(RunMetrics(**data))
        
        # Sort by timestamp
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return runs
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all runs."""
        runs = self.get_all_runs()
        
        if not runs:
            return {"total_runs": 0}
        
        return {
            "total_runs": len(runs),
            "avg_grounding_accuracy": sum(r.grounding_accuracy for r in runs) / len(runs),
            "avg_hallucination_rate": sum(r.hallucination_rate for r in runs) / len(runs),
            "avg_coherence_score": sum(r.structural_coherence for r in runs) / len(runs),
            "avg_generation_time": sum(r.generation_time_seconds for r in runs) / len(runs),
            "total_pages_generated": sum(r.total_pages for r in runs),
        }


# Global logger instance
_metrics_logger: Optional[MetricsLogger] = None


def get_logger(output_dir: str = "outputs/metrics") -> MetricsLogger:
    """Get or create the global metrics logger."""
    global _metrics_logger
    
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger(output_dir)
    
    return _metrics_logger


if __name__ == "__main__":
    # Test metrics logging
    print("Testing metrics logger...")
    
    logger = MetricsLogger("outputs/metrics_test")
    
    # Simulate a run
    run_id = logger.start_run("Test Topic", "mistral-7b")
    
    logger.log_generation(sections=5, words=2500)
    logger.log_grounding(accuracy=0.95, total_claims=40, grounded_claims=38)
    logger.log_hallucination(rate=0.02, total_citations=15, valid_citations=15)
    logger.log_coherence(score=0.92, section_scores=[0.90, 0.95, 0.88, 0.94, 0.93])
    logger.log_redundancy(rate=0.05, avg_similarity=0.15)
    logger.log_revision(iterations=1, initial_score=6.5, final_score=8.5)
    
    run = logger.end_run()
    
    print(f"\nRun completed: {run.run_id}")
    print(f"  Grounding: {run.grounding_accuracy:.1%}")
    print(f"  Hallucination: {run.hallucination_rate:.1%}")
    print(f"  Coherence: {run.structural_coherence:.2f}")
