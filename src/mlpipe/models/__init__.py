"""Model candidates and registries for MLPipe."""

from mlpipe.models.classification import get_classification_candidates
from mlpipe.models.regression import get_regression_candidates
from mlpipe.models.registry import ModelCandidate
from mlpipe.models.selection import get_candidates_for_task

__all__ = [
    "ModelCandidate",
    "get_classification_candidates",
    "get_regression_candidates",
    "get_candidates_for_task",
]
