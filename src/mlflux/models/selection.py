"""Model selection and retrieval helpers."""

from typing import List

from mlflux.models.classification import get_classification_candidates
from mlflux.models.regression import get_regression_candidates
from mlflux.models.registry import ModelCandidate


def get_candidates_for_task(task_type: str, mode: str = "balanced") -> List[ModelCandidate]:
    """
    Retrieve appropriate candidate models for a given task and training mode.
    """
    if task_type == "classification":
        candidates = get_classification_candidates()
    elif task_type == "regression":
        candidates = get_regression_candidates()
    else:
        raise ValueError(f"Unknown task type '{task_type}'")

    if mode == "fast":
        # For fast mode, we still evaluate all models or the fastest subset
        return candidates
    return candidates
