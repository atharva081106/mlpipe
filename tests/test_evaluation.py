"""Tests for evaluation metrics and leaderboard ranking."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor

from mlpipe.evaluation.evaluator import evaluate_pipeline_on_test, build_leaderboard
from mlpipe.evaluation.metrics import select_primary_metric
from mlpipe.tuning.search import TuningResult


def test_select_primary_metric():
    y_reg = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert select_primary_metric("regression", y_reg) == "r2"

    y_imbalanced = pd.Series(["No"] * 90 + ["Yes"] * 10)
    assert select_primary_metric("classification", y_imbalanced) == "f1_weighted"

    y_balanced = pd.Series(["No"] * 50 + ["Yes"] * 50)
    assert select_primary_metric("classification", y_balanced) == "roc_auc"


def test_evaluate_classification_pipeline():
    X_test = pd.DataFrame({"feat": [1, 2, 3, 4, 5, 6]})
    y_test = pd.Series(["A", "A", "A", "B", "B", "B"])

    pipe = Pipeline([("estimator", DummyClassifier(strategy="most_frequent"))])
    pipe.fit(X_test, y_test)

    metrics = evaluate_pipeline_on_test(pipe, X_test, y_test, task_type="classification")
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 0.5


def test_leaderboard_ranked_by_cv_score():
    X_test = pd.DataFrame({"feat": [1, 2, 3, 4]})
    y_test = pd.Series(["A", "B", "A", "B"])

    pipe1 = Pipeline([("estimator", DummyClassifier(strategy="most_frequent"))])
    pipe1.fit(X_test, y_test)
    pipe2 = Pipeline([("estimator", DummyClassifier(strategy="stratified", random_state=42))])
    pipe2.fit(X_test, y_test)

    tuning_results = [
        TuningResult(
            candidate_name="ModelLowCV",
            best_pipeline=pipe1,
            best_params={},
            best_cv_score=0.60,
            training_time_s=1.0,
        ),
        TuningResult(
            candidate_name="ModelHighCV",
            best_pipeline=pipe2,
            best_params={},
            best_cv_score=0.85,
            training_time_s=1.2,
        ),
    ]

    lb = build_leaderboard(tuning_results, X_test, y_test, task_type="classification", primary_metric="f1_weighted")

    # Winner must be ModelHighCV because ranking is strictly CV-based!
    assert lb[0]["model"] == "ModelHighCV"
    assert lb[0]["cv_score"] == 0.85
    assert lb[1]["model"] == "ModelLowCV"
    assert lb[1]["cv_score"] == 0.60
