"""Tests for data profiling module."""

import numpy as np
import pandas as pd
import pytest

from mlflux.data.profiling import profile_dataset, detect_column_type


def test_profiling_types(sample_classification_df: pd.DataFrame):
    profile = profile_dataset(sample_classification_df)
    assert profile.num_rows == 120
    assert profile.num_cols == 6
    assert profile.total_missing_values == 0

    num_col = profile.get_column("numeric_feat")
    assert num_col is not None
    assert num_col.detected_type == "numeric"
    assert num_col.min_val is not None
    assert num_col.max_val is not None
    assert num_col.mean_val is not None
    assert num_col.median_val is not None

    cat_col = profile.get_column("cat_feat")
    assert cat_col is not None
    assert cat_col.detected_type == "categorical"
    assert len(cat_col.top_values) > 0


def test_profiling_flags():
    df = pd.DataFrame({
        "constant_col": [1, 1, 1, 1, 1],
        "mostly_missing": [1.0, np.nan, np.nan, np.nan, np.nan],
        "id_col": [101, 102, 103, 104, 105],
        "normal_cat": ["A", "B", "A", "B", "A"],
    })
    profile = profile_dataset(df)

    const_c = profile.get_column("constant_col")
    assert any("Constant" in f for f in const_c.flags)

    missing_c = profile.get_column("mostly_missing")
    assert any("High missingness" in f for f in missing_c.flags)

    id_c = profile.get_column("id_col")
    assert any("Suspicious unique" in f or "Possible identifier" in f for f in id_c.flags)


def test_profiling_datetime_detection():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "timestamp_col": dates.astype(str),
        "val": range(10),
    })
    profile = profile_dataset(df)
    dt_col = profile.get_column("timestamp_col")
    assert dt_col.detected_type == "datetime"
    assert dt_col.min_date is not None
    assert dt_col.max_date is not None
