"""
Pipeline results and reporting structures.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PipelineResult:
    """Encapsulates the complete outcome of an MLPipe pipeline run."""

    run_id: str
    target_column: str
    task_type: str
    best_model_name: str
    primary_metric: str
    best_cv_score: float
    test_score: float
    leaderboard: List[Dict[str, Any]]
    test_metrics: Dict[str, Any]
    feature_importance: List[Dict[str, Any]]
    artifacts_dir: Path
    elapsed_time_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary representation."""
        data = asdict(self)
        data["artifacts_dir"] = str(self.artifacts_dir)
        return data

    @property
    def best_model(self) -> str:
        """Convenience alias for best model name."""
        return self.best_model_name

    @property
    def metrics(self) -> Dict[str, Any]:
        """Convenience alias for test metrics."""
        return self.test_metrics
