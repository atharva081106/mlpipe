"""
Preprocessing pipeline builder and column classifier for MLPipe.

Constructs unified sklearn ColumnTransformer structures with leakage-free feature processing.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from mlpipe.core.exceptions import PreprocessingError
from mlpipe.data.profiling import detect_column_type
from mlpipe.preprocessing.categorical import build_categorical_transformer
from mlpipe.preprocessing.datetime import DatetimeFeatureExtractor
from mlpipe.preprocessing.numeric import build_numeric_transformer
from mlpipe.utils.logging import get_logger

logger = get_logger("preprocessing")


@dataclass
class ColumnAssignments:
    """Summary of column assignments across transformation types."""

    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    dropped: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "datetime": self.datetime,
            "dropped": self.dropped,
        }


def classify_columns(df: pd.DataFrame) -> ColumnAssignments:
    """
    Categorize columns into numeric, categorical, datetime, or dropped.

    Drops:
    - Constant columns (<= 1 unique value)
    - Columns with > 90% missing values
    - Obvious unique ID / index columns (all unique string/integer identifiers)
    """
    n_rows = len(df)
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    dropped_cols: List[str] = []

    id_pattern = re.compile(r"(^id$|_id$|^id_|^index$|^guid$|^uuid$)", re.IGNORECASE)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        n_unique = non_null.nunique()

        # 1. Constant column check
        if n_unique <= 1:
            dropped_cols.append(col)
            logger.info("Dropping constant column '%s' (unique values: %d)", col, n_unique)
            continue

        # 2. Extreme missingness check (> 90%)
        if series.isna().mean() > 0.90:
            dropped_cols.append(col)
            logger.info("Dropping column '%s' with >90%% missing values", col)
            continue

        # 3. ID / index column heuristic
        if n_unique == n_rows and (id_pattern.search(str(col)) or series.dtype == object):
            dropped_cols.append(col)
            logger.info("Dropping identifier column '%s' (100%% unique values)", col)
            continue

        detected = detect_column_type(series)
        if detected == "numeric":
            numeric_cols.append(col)
        elif detected == "datetime":
            datetime_cols.append(col)
        else:  # categorical or boolean
            categorical_cols.append(col)

    if not numeric_cols and not categorical_cols and not datetime_cols:
        raise PreprocessingError(
            "No usable feature columns remain after filtering constant/ID/missing columns.",
            "Verify that your dataset contains informative feature columns with variation."
        )

    return ColumnAssignments(
        numeric=numeric_cols,
        categorical=categorical_cols,
        datetime=datetime_cols,
        dropped=dropped_cols,
    )


def build_preprocessor(
    assignments: ColumnAssignments,
    with_scaling: bool = True,
) -> ColumnTransformer:
    """
    Construct an unfitted ColumnTransformer based on column assignments.

    Args:
        assignments: ColumnAssignments partitioning features.
        with_scaling: Whether to apply StandardScaler to numeric features.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    transformers = []

    if assignments.numeric:
        num_pipeline = build_numeric_transformer(with_scaling=with_scaling)
        transformers.append(("numeric", num_pipeline, assignments.numeric))

    if assignments.categorical:
        cat_pipeline = build_categorical_transformer()
        transformers.append(("categorical", cat_pipeline, assignments.categorical))

    if assignments.datetime:
        dt_pipeline = Pipeline([
            ("extractor", DatetimeFeatureExtractor())
        ])
        transformers.append(("datetime", dt_pipeline, assignments.datetime))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.
    """
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception as e:
        logger.debug("Failed to get feature names via get_feature_names_out: %s", e)
        # Fallback: inspect transformers
        names: List[str] = []
        for name, trans, cols in fitted_preprocessor.transformers_:
            if name == "remainder" or trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    names.extend(list(trans.get_feature_names_out(cols)))
                except Exception:
                    names.extend(list(cols))
            else:
                names.extend(list(cols))
        return names
