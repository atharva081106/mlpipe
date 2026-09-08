"""
Metric definitions and primary metric selection logic for MLFlux.
"""

from typing import Dict, Optional, Tuple

import pandas as pd


def select_primary_metric(task_type: str, y_train: pd.Series) -> str:
    """
    Select the optimal primary optimization metric.

    Documented Rationale:
    - Regression:
        Uses 'r2' (Coefficient of Determination) to measure variance explained.
    - Classification:
        - Checks class distribution in y_train.
        - If binary and balanced (minority >= 20%): uses 'roc_auc' or 'f1' depending on class balance.
        - If imbalanced (minority < 20%): uses 'f1_weighted' to penalize poor minority performance
          rather than misleading accuracy.
        - If multiclass (> 2 classes): uses 'f1_weighted' to account for class representation.
    """
    if task_type == "regression":
        return "r2"

    val_counts = y_train.value_counts(normalize=True)
    n_classes = len(val_counts)

    if n_classes == 2:
        minority_prop = float(val_counts.min())
        if minority_prop < 0.20:
            # Imbalanced binary
            return "f1_weighted"
        return "roc_auc"
    else:
        # Multiclass
        return "f1_weighted"


def get_display_metric_name(metric_key: str) -> str:
    """Map sklearn scoring key to friendly display name."""
    mapping = {
        "f1_weighted": "F1 (Weighted)",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "r2": "R²",
        "neg_mean_squared_error": "MSE",
        "neg_mean_absolute_error": "MAE",
    }
    return mapping.get(metric_key, metric_key.upper())
