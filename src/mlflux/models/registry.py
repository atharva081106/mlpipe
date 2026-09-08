"""Model candidate representations and specifications."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type


@dataclass
class ModelCandidate:
    """Specification of an ML estimator candidate."""

    name: str
    estimator_factory: Callable[..., Any]
    task: str  # "classification" or "regression"
    supports_feature_importance: bool = True
    importance_type: str = "none"  # "tree", "linear", or "none"
    requires_scaling: bool = False
    param_grid_fast: Dict[str, Any] = field(default_factory=dict)
    param_grid_balanced: Dict[str, Any] = field(default_factory=dict)
    param_grid_thorough: Dict[str, Any] = field(default_factory=dict)

    def create_estimator(self, random_seed: int = 42, **kwargs) -> Any:
        """Instantiate estimator with seed if supported."""
        return self.estimator_factory(random_seed=random_seed, **kwargs)

    def get_search_space(self, mode: str) -> Dict[str, Any]:
        """Return the hyperparameter search space for the given training mode."""
        mode = mode.lower()
        if mode == "fast":
            return self.param_grid_fast
        elif mode == "thorough":
            return self.param_grid_thorough
        return self.param_grid_balanced
