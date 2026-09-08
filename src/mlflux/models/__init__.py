"""Model candidates and registries for MLFlux."""

from mlflux.models.classification import get_classification_candidates
from mlflux.models.regression import get_regression_candidates
from mlflux.models.registry import ModelCandidate
from mlflux.models.selection import get_candidates_for_task

__all__ = [
    "ModelCandidate",
    "get_classification_candidates",
    "get_regression_candidates",
    "get_candidates_for_task",
]
