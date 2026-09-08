"""Categorical feature preprocessing pipelines."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_categorical_transformer() -> Pipeline:
    """
    Build a preprocessing pipeline for categorical features.

    Applies most-frequent imputation followed by one-hot encoding with unseen category handling.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
