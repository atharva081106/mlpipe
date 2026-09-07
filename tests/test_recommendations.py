import pytest
import pandas as pd
import numpy as np

from mlpipe.core.recommendations import (
    recommend_target_column,
    recommend_split_strategy,
    recommend_models_for_task,
    recommend_hyperparameter_overrides,
)


def test_recommend_target_by_keyword():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, 45, 30, 35, 40],
        "churn": [0, 1, 0, 0, 1],
    })
    col, reason = recommend_target_column(df)
    assert col == "churn"
    assert "churn" in reason or "target" in reason.lower() or "binary" in reason.lower()


def test_recommend_target_by_position():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_b": [10, 20, 30, 40, 50],
        "reading": [100.0, 200.0, 300.0, 400.0, 500.0],
    })
    col, reason = recommend_target_column(df)
    assert col == "reading"
    assert "position" in reason.lower() or "continuous" in reason.lower()


def test_recommend_target_avoids_id():
    df = pd.DataFrame({
        "user_uuid": ["a", "b", "c", "d", "e"],
        "feature_1": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
    })
    col, reason = recommend_target_column(df)
    assert col == "target"


def test_recommend_split_strategy():
    small = recommend_split_strategy(50, "classification")
    assert small["test_size"] == 0.25
    assert small["is_stratified"] is True

    standard = recommend_split_strategy(5000, "regression")
    assert standard["test_size"] == 0.20
    assert standard["is_stratified"] is False

    large = recommend_split_strategy(200_000, "classification")
    assert large["test_size"] == 0.10


def test_recommend_models_for_task():
    cands = ["HistGradientBoosting Classifier", "Random Forest Classifier", "Logistic Regression"]
    rec = recommend_models_for_task("classification", 1000, cands)
    assert rec["recommended_choice"] == "A"
    assert "HistGradientBoosting Classifier" in rec["candidate_notes"]
    assert "⭐" in rec["candidate_notes"]["HistGradientBoosting Classifier"]


def test_recommend_hyperparameter_overrides_rf():
    params = {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
    recs = recommend_hyperparameter_overrides("Random Forest Classifier", params)
    assert "n_estimators" in recs
    assert recs["n_estimators"]["recommended"] == 150
    assert "max_depth" in recs
    assert recs["max_depth"]["recommended"] == 12


def test_recommend_hyperparameter_overrides_hgb():
    params = {"learning_rate": 0.1, "max_iter": 100, "l2_regularization": 0.0}
    recs = recommend_hyperparameter_overrides("HistGradientBoosting Regressor", params)
    assert "learning_rate" in recs
    assert recs["learning_rate"]["recommended"] == 0.05
    assert recs["max_iter"]["recommended"] == 150


def test_recommend_target_graduate_admissions():
    # Exact structure from user's screenshot
    df = pd.DataFrame({
        "Serial_No.": [1, 2, 3, 4, 5],
        "GRE_Score": [337.0, 324.0, 316.0, 322.0, 314.0],
        "TOEFL_Score": [118.0, 107.0, 104.0, 110.0, 103.0],
        "University_Rating": [4, 3, 2, 3, 2],
        "SOP": [4.5, 4.0, 3.0, 3.5, 2.0],
        "LOR": [4.5, 3.5, 2.5, 2.5, 3.0],
        "CGPA": [9.65, 8.87, 8.0, 8.67, 8.21],
        "Research": ["Yes", "No", "No", "Yes", "No"],
        "Chance_of_Admit": [0.92, 0.76, 0.72, 0.80, 0.65],
    })
    col, reason = recommend_target_column(df)
    assert col == "Chance_of_Admit"
    assert "admit" in reason.lower() or "chance" in reason.lower() or "probability" in reason.lower()

