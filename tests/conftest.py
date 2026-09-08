"""Pytest fixtures and sample data for MLFlux tests."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """Fixture providing a clean tabular classification DataFrame."""
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(n)],
        "numeric_feat": np.random.normal(50, 15, n),
        "skewed_feat": np.random.exponential(10, n),
        "cat_feat": np.random.choice(["TypeA", "TypeB", "TypeC"], n),
        "bool_feat": np.random.choice([True, False], n),
        "target": np.random.choice(["Class1", "Class2"], n, p=[0.6, 0.4]),
    })


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    """Fixture providing a clean tabular regression DataFrame."""
    np.random.seed(42)
    n = 120
    x1 = np.random.uniform(10, 100, n)
    x2 = np.random.choice(["Group1", "Group2", "Group3"], n)
    group_bonus = {"Group1": 10.0, "Group2": 25.0, "Group3": 50.0}
    y = 2.5 * x1 + np.array([group_bonus[g] for g in x2]) + np.random.normal(0, 5, n)
    return pd.DataFrame({
        "feature_num": x1,
        "feature_cat": x2,
        "target_num": y,
    })


@pytest.fixture
def sample_csv_file(tmp_path: Path, sample_classification_df: pd.DataFrame) -> Path:
    """Fixture providing a temporary CSV file path."""
    csv_path = tmp_path / "test_data.csv"
    sample_classification_df.to_csv(csv_path, index=False)
    return csv_path
