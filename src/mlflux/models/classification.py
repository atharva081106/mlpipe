"""Classification candidate models and parameter spaces."""

from typing import List

from scipy.stats import randint, uniform
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from mlflux.models.registry import ModelCandidate


def get_classification_candidates() -> List[ModelCandidate]:
    """Return all registered classification candidates."""

    return [
        # 1. Logistic Regression
        ModelCandidate(
            name="Logistic Regression",
            estimator_factory=lambda random_seed=42, **kwargs: LogisticRegression(
                random_state=random_seed, max_iter=1000, **kwargs
            ),
            task="classification",
            supports_feature_importance=True,
            importance_type="linear",
            requires_scaling=True,
            param_grid_fast={
                "C": [0.1, 1.0, 10.0],
            },
            param_grid_balanced={
                "C": uniform(0.01, 10.0),
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
            param_grid_thorough={
                "C": uniform(0.001, 50.0),
                "penalty": ["l2"],
                "solver": ["lbfgs", "saga"],
            },
        ),

        # 2. Random Forest
        ModelCandidate(
            name="Random Forest",
            estimator_factory=lambda random_seed=42, **kwargs: RandomForestClassifier(
                random_state=random_seed, n_jobs=-1, **kwargs
            ),
            task="classification",
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
                "max_features": ["sqrt", "log2", None],
            },
        ),

        # 3. HistGradientBoosting
        ModelCandidate(
            name="HistGradientBoosting",
            estimator_factory=lambda random_seed=42, **kwargs: HistGradientBoostingClassifier(
                random_state=random_seed, **kwargs
            ),
            task="classification",
            supports_feature_importance=False,  # sklearn's HistGradientBoosting doesn't expose feature_importances_ natively
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

        # 4. Decision Tree
        ModelCandidate(
            name="Decision Tree",
            estimator_factory=lambda random_seed=42, **kwargs: DecisionTreeClassifier(
                random_state=random_seed, **kwargs
            ),
            task="classification",
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
                "criterion": ["gini", "entropy", "log_loss"],
            },
        ),

        # 5. K-Nearest Neighbors
        ModelCandidate(
            name="K-Nearest Neighbors",
            estimator_factory=lambda random_seed=42, **kwargs: KNeighborsClassifier(
                **kwargs
            ),
            task="classification",
            supports_feature_importance=False,
            importance_type="none",
            requires_scaling=True,
            param_grid_fast={
                "n_neighbors": [3, 5, 7],
            },
            param_grid_balanced={
                "n_neighbors": randint(3, 15),
                "weights": ["uniform"],
                "metric": ["euclidean", "manhattan"],
            },
            param_grid_thorough={
                "n_neighbors": randint(2, 25),
                "weights": ["uniform"],
                "metric": ["euclidean", "manhattan", "minkowski"],
                "p": [1, 2],
            },
        ),

    ]
