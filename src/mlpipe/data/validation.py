"""
Data validation and automatic task detection for MLPipe.

Provides rigorous pre-training checks separating fatal errors from warnings,
and infers task type (classification vs. regression) from target distribution.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mlpipe.core.exceptions import ValidationError
from mlpipe.data.ingestion import Dataset


def detect_task(target_series: pd.Series) -> str:
    """
    Automatically determine if the problem is classification or regression.

    Heuristic:
    - Object, string, category, boolean dtypes -> classification.
    - Low-cardinality integers (<= 10 unique or <= 2% of rows) -> classification.
    - Floating point numbers with high cardinality -> regression.
    """
    clean_target = target_series.dropna()
    n_unique = clean_target.nunique()
    n_total = len(clean_target)

    if n_total == 0 or n_unique <= 1:
        return "classification"

    if pd.api.types.is_bool_dtype(clean_target):
        return "classification"

    if clean_target.dtype == object or pd.api.types.is_string_dtype(clean_target):
        return "classification"

    if pd.api.types.is_float_dtype(clean_target):
        # Float targets are regression unless they only contain a tiny set of integers like 0.0, 1.0
        unique_vals = clean_target.unique()
        if len(unique_vals) <= 5 and all(float(v).is_integer() for v in unique_vals):
            return "classification"
        return "regression"

    if pd.api.types.is_integer_dtype(clean_target):
        # If very few unique values (e.g. 2 to 10 classes) and small ratio compared to rows -> classification
        if n_unique <= 10 or (n_unique <= 20 and (n_unique / n_total) < 0.05):
            return "classification"
        return "regression"

    return "classification"


@dataclass
class ValidationReport:
    """Detailed outcome of pre-training dataset validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detected_task: str = "classification"
    target_info: Dict[str, Any] = field(default_factory=dict)
    can_train: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def raise_if_invalid(self) -> None:
        """Raise ValidationError if any fatal check failed."""
        if not self.can_train or not self.is_valid:
            error_details = "\n• " + "\n• ".join(self.errors)
            raise ValidationError(
                f"Dataset validation failed with {len(self.errors)} fatal error(s):{error_details}",
                "Fix the fatal issues indicated above before proceeding with model training."
            )


def validate_dataset(
    dataset: Union[Dataset, pd.DataFrame],
    target_column: str,
    task_override: Optional[str] = None,
    test_size: float = 0.20,
    cv_folds: int = 5,
) -> ValidationReport:
    """
    Perform comprehensive pre-training validation.

    Checks:
    - Target existence
    - Target usable values and missingness
    - Target variation (non-constant)
    - Minimum sample count
    - Minimum class counts (for classification)
    - Available feature columns
    - Train/test and CV fold feasibility
    - Class imbalance
    """
    df = dataset.df if isinstance(dataset, Dataset) else dataset
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Target column existence
    if target_column not in df.columns:
        errors.append(
            f"Target column '{target_column}' does not exist in the dataset. "
            f"Available columns: {list(df.columns)}"
        )
        return ValidationReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            detected_task="unknown",
            target_info={},
            can_train=False,
        )

    target = df[target_column]
    n_rows = len(df)
    target_missing = int(target.isna().sum())
    target_valid_rows = n_rows - target_missing

    # 2. Target missingness
    if target_valid_rows == 0:
        errors.append(f"Target column '{target_column}' contains only missing (NaN/null) values.")
    elif target_missing > 0:
        missing_pct = round((target_missing / n_rows) * 100, 2)
        if missing_pct > 50.0:
            errors.append(
                f"Target column '{target_column}' has {missing_pct}% missing values (more than 50%)."
            )
        else:
            warnings.append(
                f"Target column '{target_column}' contains {target_missing} missing values ({missing_pct}%). "
                f"These {target_missing} rows will be dropped prior to training."
            )

    # 3. Available feature columns
    feature_cols = [c for c in df.columns if c != target_column]
    if len(feature_cols) == 0:
        errors.append("Dataset contains only the target column; at least one feature column is required.")

    # 4. Minimum rows check
    if target_valid_rows < 10:
        errors.append(
            f"Dataset has only {target_valid_rows} usable rows. MLPipe requires at least 10 rows."
        )
    elif target_valid_rows < 50:
        warnings.append(
            f"Dataset has only {target_valid_rows} usable rows. Results may have high variance."
        )

    # 5. Task detection or override
    inferred_task = detect_task(target)
    if task_override and task_override != "auto":
        task = task_override.lower()
    else:
        task = inferred_task

    target_info: Dict[str, Any] = {
        "column": target_column,
        "task": task,
        "inferred_task": inferred_task,
        "total_rows": n_rows,
        "usable_rows": target_valid_rows,
        "missing_count": target_missing,
    }

    # 6. Task-specific checks
    clean_target = target.dropna()
    unique_targets = clean_target.unique()
    n_classes = len(unique_targets)
    target_info["unique_values"] = n_classes

    if task == "classification":
        if n_classes < 2:
            errors.append(
                f"Target column '{target_column}' has only {n_classes} unique class. "
                "Classification requires at least 2 distinct classes."
            )
        else:
            target_info["classes"] = [str(c) for c in unique_targets[:10]]
            # Check class distribution
            val_counts = clean_target.value_counts()
            min_class_count = int(val_counts.min())
            min_class_pct = round((min_class_count / len(clean_target)) * 100, 2)
            target_info["min_class_count"] = min_class_count
            target_info["min_class_pct"] = min_class_pct

            # Check if smallest class has enough samples for train/test and CV
            if min_class_count < cv_folds:
                warnings.append(
                    f"Minority class '{val_counts.idxmin()}' has only {min_class_count} samples, "
                    f"which is fewer than cv_folds ({cv_folds}). Stratified folds may be adjusted."
                )

            if min_class_pct < 15.0:
                warnings.append(
                    f"Target is imbalanced: minority class represents only {min_class_pct}% of samples."
                )

    elif task == "regression":
        if n_classes <= 1:
            errors.append(
                f"Target column '{target_column}' has only 1 distinct numeric value. "
                "Regression requires variation in the target."
            )
        else:
            try:
                target_numeric = pd.to_numeric(clean_target, errors="raise")
                target_info["mean"] = float(round(target_numeric.mean(), 4))
                target_info["std"] = float(round(target_numeric.std(), 4))
            except Exception:
                errors.append(
                    f"Target column '{target_column}' contains non-numeric values that cannot be used for regression."
                )

    # 7. Train/Test feasibility
    test_rows = int(target_valid_rows * test_size)
    train_rows = target_valid_rows - test_rows
    if train_rows < 5:
        errors.append(f"Training set would have only {train_rows} rows; increase dataset size or decrease test_size.")
    if test_rows < 2:
        errors.append(f"Test set would have only {test_rows} rows; evaluation cannot be reliably performed.")

    # 8. Feature quality warnings
    for col in feature_cols:
        col_series = df[col]
        if col_series.nunique(dropna=True) <= 1:
            warnings.append(f"Feature column '{col}' is constant (<=1 unique value) and will be dropped.")
        elif col_series.isna().mean() > 0.90:
            warnings.append(f"Feature column '{col}' has >90% missing values and will be dropped.")

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        detected_task=task,
        target_info=target_info,
        can_train=is_valid,
    )
