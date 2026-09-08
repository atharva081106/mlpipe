"""Tests for data validation and task detection."""

import pandas as pd
import pytest

from mlflux.core.exceptions import ValidationError
from mlflux.data.validation import detect_task, validate_dataset


def test_detect_task_classification():
    s_binary = pd.Series(["yes", "no", "yes", "yes", "no"] * 20)
    assert detect_task(s_binary) == "classification"

    s_multiclass = pd.Series(["A", "B", "C"] * 20)
    assert detect_task(s_multiclass) == "classification"

    s_int_classes = pd.Series([0, 1] * 30)
    assert detect_task(s_int_classes) == "classification"


def test_detect_task_regression():
    s_continuous = pd.Series([10.5 + i * 2.3 for i in range(100)])
    assert detect_task(s_continuous) == "regression"


def test_validation_valid(sample_classification_df: pd.DataFrame):
    report = validate_dataset(sample_classification_df, target_column="target")
    assert report.is_valid
    assert report.can_train
    assert len(report.errors) == 0
    assert report.detected_task == "classification"


def test_validation_missing_target_column(sample_classification_df: pd.DataFrame):
    report = validate_dataset(sample_classification_df, target_column="non_existent")
    assert not report.is_valid
    assert not report.can_train
    assert any("does not exist" in err for err in report.errors)
    with pytest.raises(ValidationError):
        report.raise_if_invalid()


def test_validation_constant_target():
    df = pd.DataFrame({
        "feat": range(50),
        "target": ["SameClass"] * 50,
    })
    report = validate_dataset(df, target_column="target")
    assert not report.is_valid
    assert any("Classification requires at least 2 distinct classes" in err for err in report.errors)
