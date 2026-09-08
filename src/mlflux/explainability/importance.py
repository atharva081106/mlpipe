"""
Feature importance extraction and model explainability for MLFlux.

Recovers post-transformation feature names to provide transparent explanations
for tree-based and linear models.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.pipeline import Pipeline

from mlflux.preprocessing.builder import get_transformed_feature_names
from mlflux.utils.logging import get_logger

logger = get_logger("explainability")


def extract_feature_importance(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """
    Extract feature importance or coefficients from a fitted sklearn Pipeline.

    Uses post-transformation feature names from the preprocessor step.
    Returns sorted list of features with importance scores.
    """
    if "preprocessor" not in pipeline.named_steps or "estimator" not in pipeline.named_steps:
        return []

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["estimator"]

    feature_names = get_transformed_feature_names(preprocessor)
    n_features = len(feature_names)

    scores: Optional[np.ndarray] = None

    # 1. Tree-based models (Random Forest, Decision Tree)
    if hasattr(estimator, "feature_importances_"):
        raw_imp = estimator.feature_importances_
        if len(raw_imp) == n_features:
            scores = raw_imp

    # 2. Linear models (LogisticRegression, Ridge)
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim == 2:
            # Multiclass: average absolute coefficients across classes
            scores = np.mean(np.abs(coef), axis=0)
        else:
            scores = np.abs(coef)

        if len(scores) != n_features:
            scores = None

    if scores is None or len(scores) == 0:
        return []

    # Pair with feature names and sort descending
    paired = [
        {"feature": str(name), "importance": float(round(abs(score), 4))}
        for name, score in zip(feature_names, scores)
    ]
    paired.sort(key=lambda x: x["importance"], reverse=True)

    return paired
