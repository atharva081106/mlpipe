"""
Model evaluation engine for MLPipe.

Calculates unbiased test-set metrics and compiles the model comparison leaderboard.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from mlpipe.evaluation.metrics import get_display_metric_name
from mlpipe.tuning.search import TuningResult
from mlpipe.utils.logging import get_logger

logger = get_logger("evaluation")


def evaluate_pipeline_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> Dict[str, Any]:
    """
    Evaluate a fitted pipeline on held-out test data.

    All metrics are directly computed from actual model predictions.
    """
    y_pred = pipeline.predict(X_test)
    metrics: Dict[str, Any] = {}

    if task_type == "classification":
        acc = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        metrics["accuracy"] = round(acc, 4)
        metrics["balanced_accuracy"] = round(bal_acc, 4)
        metrics["precision"] = round(prec, 4)
        metrics["recall"] = round(rec, 4)
        metrics["f1_weighted"] = round(f1, 4)

        # Calculate ROC-AUC if predict_proba is supported
        if hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)
                classes = np.unique(y_test)
                if len(classes) == 2:
                    auc = float(roc_auc_score(y_test, y_prob[:, 1]))
                    metrics["roc_auc"] = round(auc, 4)
                elif len(classes) > 2:
                    auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
                    metrics["roc_auc"] = round(auc, 4)
            except Exception as e:
                logger.debug("ROC-AUC computation skipped: %s", e)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

    else:  # regression
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(math.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        metrics["mae"] = round(mae, 4)
        metrics["mse"] = round(mse, 4)
        metrics["rmse"] = round(rmse, 4)
        metrics["r2"] = round(r2, 4)

    return metrics


def build_leaderboard(
    tuning_results: List[TuningResult],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    primary_metric: str,
) -> List[Dict[str, Any]]:
    """
    Compile and rank candidate models into a leaderboard.

    MODELS ARE RANKED STRICTLY BY CV SCORE ON TRAINING DATA.
    Test scores are calculated for reporting but never used for model selection.
    """
    leaderboard = []

    for res in tuning_results:
        if res.best_pipeline is None or res.error:
            leaderboard.append({
                "model": res.candidate_name,
                "cv_score": None,
                "test_score": None,
                "training_time_s": res.training_time_s,
                "status": "failed",
                "error": res.error,
                "best_params": res.best_params,
            })
            continue

        # Evaluate on test set
        test_metrics = evaluate_pipeline_on_test(res.best_pipeline, X_test, y_test, task_type)

        # Extract primary metric test value
        if task_type == "regression":
            test_score = test_metrics.get("r2", 0.0)
        else:
            test_score = test_metrics.get(primary_metric, test_metrics.get("f1_weighted", 0.0))

        leaderboard.append({
            "model": res.candidate_name,
            "cv_score": res.best_cv_score,
            "test_score": test_score,
            "training_time_s": res.training_time_s,
            "status": "success",
            "error": None,
            "best_params": res.best_params,
            "test_metrics": test_metrics,
        })

    # Sort descending by CV score (failed models with None placed at the end)
    leaderboard.sort(
        key=lambda x: x["cv_score"] if x["cv_score"] is not None else -999999.0,
        reverse=True,
    )

    return leaderboard
