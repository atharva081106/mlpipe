"""Tests for preprocessing pipeline and transformers."""

import numpy as np
import pandas as pd
import pytest

from mlpipe.preprocessing.builder import (
    build_preprocessor,
    classify_columns,
    get_transformed_feature_names,
)


def test_classify_columns_drops_constants_and_ids():
    df = pd.DataFrame({
        "row_id": range(100),
        "const_num": [42] * 100,
        "valid_num": np.random.randn(100),
        "valid_cat": np.random.choice(["A", "B", "C"], 100),
    })
    assignments = classify_columns(df)
    assert "row_id" in assignments.dropped
    assert "const_num" in assignments.dropped
    assert "valid_num" in assignments.numeric
    assert "valid_cat" in assignments.categorical


def test_preprocessing_fit_transform_and_unseen_categories():
    train_df = pd.DataFrame({
        "num": [10.0, np.nan, 30.0, 40.0],
        "cat": ["cat", "dog", "cat", "bird"],
    })
    test_df = pd.DataFrame({
        "num": [20.0, 50.0],
        "cat": ["dog", "elephant"],  # 'elephant' is an unseen category
    })

    assignments = classify_columns(train_df)
    preprocessor = build_preprocessor(assignments, with_scaling=True)

    # Fit on train ONLY
    X_train_trans = preprocessor.fit_transform(train_df)
    assert X_train_trans.shape[0] == 4
    assert not np.isnan(X_train_trans).any()

    # Transform test (must handle unseen category gracefully without error)
    X_test_trans = preprocessor.transform(test_df)
    assert X_test_trans.shape[0] == 2
    assert not np.isnan(X_test_trans).any()

    # Feature names
    names = get_transformed_feature_names(preprocessor)
    assert len(names) == X_train_trans.shape[1]
    assert any("cat" in n for n in names)
