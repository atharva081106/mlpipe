"""
Hyperparameter search and model training engine.

Runs cross-validation and hyperparameter optimization strictly on training data,
ensuring complete data leakage prevention.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from mlflux.models.registry import ModelCandidate
from mlflux.tuning.spaces import get_search_iterations
from mlflux.utils.logging import get_logger

logger = get_logger("tuning")


@dataclass
class TuningResult:
    """Outcome of tuning a single candidate model."""

    candidate_name: str
    best_pipeline: Optional[Pipeline]
    best_params: Dict[str, Any]
    best_cv_score: float
    training_time_s: float
    error: Optional[str] = None


def tune_candidate(
    candidate: ModelCandidate,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    primary_metric: str,
    cv_folds: int = 5,
    mode: str = "balanced",
    random_seed: int = 42,
) -> TuningResult:
    """
    Train and tune a candidate model using RandomizedSearchCV on X_train.

    Never touches test data. Preprocessing is encapsulated inside the Pipeline
    so each CV fold fits preprocessing strictly on that fold's training portion.
    """
    t_start = time.perf_counter()

    try:
        # 1. Instantiate base estimator
        base_estimator = candidate.create_estimator(random_seed=random_seed)

        # 2. Construct complete pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", base_estimator),
        ])

        # 3. Determine CV splitter
        n_samples = len(X_train)
        effective_folds = min(cv_folds, n_samples)

        if candidate.task == "classification":
            val_counts = y_train.value_counts()
            min_class = int(val_counts.min()) if len(val_counts) > 0 else 2
            effective_folds = max(2, min(effective_folds, min_class))
            cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_seed)
        else:
            effective_folds = max(2, effective_folds)
            cv = KFold(n_splits=effective_folds, shuffle=True, random_state=random_seed)

        # 4. Get search space and format for Pipeline
        raw_space = candidate.get_search_space(mode)
        param_distributions = {
            f"estimator__{k}": v for k, v in raw_space.items()
        }

        n_iter = get_search_iterations(mode)

        if param_distributions:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=primary_metric,
                random_state=random_seed,
                n_jobs=None,
                refit=True,
                error_score=np.nan,
            )

            search.fit(X_train, y_train)

            best_pipe = search.best_estimator_
            best_cv = float(search.best_score_)
            best_params = {
                k.replace("estimator__", ""): (
                    float(round(v, 4)) if isinstance(v, (float, np.floating)) else v
                )
                for k, v in search.best_params_.items()
            }
        else:
            # If no params to tune, fit and cross-validate directly
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring=primary_metric, n_jobs=-1
            )
            pipeline.fit(X_train, y_train)
            best_pipe = pipeline
            best_cv = float(cv_scores.mean())
            best_params = {}

        elapsed = round(time.perf_counter() - t_start, 2)

        return TuningResult(
            candidate_name=candidate.name,
            best_pipeline=best_pipe,
            best_params=best_params,
            best_cv_score=round(best_cv, 4),
            training_time_s=elapsed,
            error=None,
        )

    except Exception as e:
        elapsed = round(time.perf_counter() - t_start, 2)
        logger.warning("Training failed for %s: %s", candidate.name, e)
        return TuningResult(
            candidate_name=candidate.name,
            best_pipeline=None,
            best_params={},
            best_cv_score=-9999.0,
            training_time_s=elapsed,
            error=str(e),
        )
