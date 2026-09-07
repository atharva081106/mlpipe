"""Regression candidate models and parameter spaces."""

from typing import List

from scipy.stats import randint, uniform
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from mlpipe.models.registry import ModelCandidate


def get_regression_candidates() -> List[ModelCandidate]:
    """Return all registered regression candidates."""

    return [
        # 1. Ridge Regression
        ModelCandidate(
            name="Ridge",
            estimator_factory=lambda random_seed=42, **kwargs: Ridge(
                random_state=random_seed, **kwargs
            ),
            task="regression",
            supports_feature_importance=True,
            importance_type="linear",
            requires_scaling=True,
            param_grid_fast={
                "alpha": [0.1, 1.0, 10.0],
            },
            param_grid_balanced={
                "alpha": uniform(0.01, 50.0),
                "solver": ["auto", "svd", "cholesky", "lsqr"],
            },
            param_grid_thorough={
                "alpha": uniform(0.001, 200.0),
                "solver": ["auto", "svd", "cholesky", "lsqr", "sag"],
            },
        ),

        # 2. Random Forest Regressor
        ModelCandidate(
            name="Random Forest Regressor",
            estimator_factory=lambda random_seed=42, **kwargs: RandomForestRegressor(
                random_state=random_seed, n_jobs=-1, **kwargs
            ),
            task="regression",
            supports_feature_importance=True,
            importance_type="tree",
            requires_scaling=False,
            param_grid_fast={
                "n_estimators": [50, 100],
                "max_depth": [5, 10, None],
            },
            param_grid_balanced={
                "n_estimators": randint(50, 200),
                "max_depth": [5, 10, 20, None],
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 6),
            },
            param_grid_thorough={
                "n_estimators": randint(50, 300),
                "max_depth": [5, 10, 20, 30, None],
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", 1.0],
            },
        ),

        # 3. HistGradientBoosting Regressor
        ModelCandidate(
            name="HistGradientBoosting Regressor",
            estimator_factory=lambda random_seed=42, **kwargs: HistGradientBoostingRegressor(
                random_state=random_seed, **kwargs
            ),
            task="regression",
            supports_feature_importance=False,
            importance_type="none",
            requires_scaling=False,
            param_grid_fast={
                "max_iter": [50, 100],
                "learning_rate": [0.05, 0.1],
            },
            param_grid_balanced={
                "max_iter": randint(50, 200),
                "learning_rate": uniform(0.01, 0.25),
                "max_depth": [3, 5, 10, None],
                "min_samples_leaf": randint(10, 40),
            },
            param_grid_thorough={
                "max_iter": randint(50, 300),
                "learning_rate": uniform(0.005, 0.3),
                "max_depth": [3, 5, 10, 20, None],
                "min_samples_leaf": randint(5, 50),
                "l2_regularization": uniform(0.0, 2.0),
            },
        ),

        # 4. Decision Tree Regressor
        ModelCandidate(
            name="Decision Tree Regressor",
            estimator_factory=lambda random_seed=42, **kwargs: DecisionTreeRegressor(
                random_state=random_seed, **kwargs
            ),
            task="regression",
            supports_feature_importance=True,
            importance_type="tree",
            requires_scaling=False,
            param_grid_fast={
                "max_depth": [3, 5, 10, None],
            },
            param_grid_balanced={
                "max_depth": [3, 5, 10, 20, None],
                "min_samples_split": randint(2, 15),
                "min_samples_leaf": randint(1, 8),
            },
            param_grid_thorough={
                "max_depth": [3, 5, 10, 20, 30, None],
                "min_samples_split": randint(2, 30),
                "min_samples_leaf": randint(1, 15),
                "criterion": ["squared_error", "friedman_mse", "absolute_error"],
            },
        ),
    ]
