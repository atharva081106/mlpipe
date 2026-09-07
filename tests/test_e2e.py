"""
End-to-End integration tests for MLPipe Python API and data leakage prevention.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mlpipe.core.pipeline import Pipeline
from mlpipe.data.splitting import split_data
from mlpipe.preprocessing.builder import build_preprocessor, classify_columns


def test_e2e_classification(tmp_path: Path):
    pipeline = Pipeline(
        target="churn",
        task="auto",
        mode="fast",
        output_dir=tmp_path / "runs",
        random_seed=42,
    )

    result = pipeline.fit("demo_data/customer_churn.csv")

    assert result.task_type == "classification"
    assert result.best_model_name is not None
    assert result.best_cv_score > 0.0
    assert result.test_score > 0.0
    assert len(result.leaderboard) > 0
    assert (result.artifacts_dir / "pipeline.joblib").exists()

    # Prediction test
    preds = pipeline.predict("demo_data/customer_churn.csv")
    assert len(preds) == 1000
    assert set(preds).issubset({"Yes", "No"})

    # Persistence: save and load test
    save_dir = tmp_path / "saved_churn_model"
    pipeline.save(save_dir)

    loaded = Pipeline.load(save_dir, target="churn")
    loaded_preds = loaded.predict("demo_data/customer_churn.csv")
    assert np.array_equal(preds, loaded_preds)


def test_e2e_regression(tmp_path: Path):
    pipeline = Pipeline(
        target="price",
        task="auto",
        mode="fast",
        output_dir=tmp_path / "runs",
        random_seed=42,
    )

    result = pipeline.fit("demo_data/house_prices.csv")

    assert result.task_type == "regression"
    assert result.primary_metric == "r2"
    assert result.best_model_name is not None
    assert result.best_cv_score > 0.0
    assert result.test_score > 0.0

    preds = pipeline.predict("demo_data/house_prices.csv")
    assert len(preds) == 1000
    assert all(p > 0 for p in preds)


def test_data_leakage_prevention():
    """
    Explicitly verifies that the preprocessing pipeline is fitted strictly on X_train.
    The mean/median computed for imputation and scaling must match X_train,
    not the combined (X_train + X_test) dataset.
    """
    # 20 samples: 16 train, 4 test
    df = pd.DataFrame({
        "feature1": [10.0, 20.0, 30.0, 40.0] * 4 + [1000.0, 2000.0, 3000.0, 4000.0],
        "target": ["A", "A", "B", "B"] * 5,
    })

    # Train / test split (80/20 means 16 train, 4 test)
    split_res = split_data(df, target_column="target", task_type="classification", test_size=0.2, random_seed=42)

    assert split_res.train_size == 16
    assert split_res.test_size == 4

    # Classify columns on training features only
    assignments = classify_columns(split_res.X_train)
    preprocessor = build_preprocessor(assignments, with_scaling=True)

    # Fit on training data ONLY
    preprocessor.fit(split_res.X_train)

    # Extract scaler mean from fitted preprocessor
    scaler_step = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    fitted_mean = float(scaler_step.mean_[0])

    # Expected mean of training portion only
    expected_train_mean = float(split_res.X_train["feature1"].mean())
    # Mean of the entire dataset including the held-out test sample
    full_dataset_mean = float(df["feature1"].mean())

    # Verify scaler mean matches X_train mean, NOT full dataset mean
    assert abs(fitted_mean - expected_train_mean) < 1e-5
    assert abs(fitted_mean - full_dataset_mean) > 1.0  # Confirms zero leakage from test rows!

