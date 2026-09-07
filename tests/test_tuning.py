"""Tests for model tuning and RandomizedSearchCV execution."""

import numpy as np
import pandas as pd

from mlpipe.models.classification import get_classification_candidates
from mlpipe.preprocessing.builder import build_preprocessor, classify_columns
from mlpipe.tuning.search import tune_candidate


def test_tune_single_candidate(sample_classification_df: pd.DataFrame):
    X = sample_classification_df.drop(columns=["target"])
    y = sample_classification_df["target"]

    assignments = classify_columns(X)
    preprocessor = build_preprocessor(assignments)

    candidates = get_classification_candidates()
    dt_cand = next(c for c in candidates if c.name == "Decision Tree")

    res = tune_candidate(
        candidate=dt_cand,
        preprocessor=preprocessor,
        X_train=X,
        y_train=y,
        primary_metric="f1_weighted",
        cv_folds=3,
        mode="fast",
        random_seed=42,
    )

    assert res.error is None
    assert res.best_pipeline is not None
    assert res.best_cv_score > 0.0
    assert res.training_time_s >= 0.0
