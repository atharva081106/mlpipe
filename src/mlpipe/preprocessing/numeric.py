"""Numeric feature preprocessing pipelines."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_numeric_transformer(with_scaling: bool = True) -> Pipeline:
    """
    Build a preprocessing pipeline for numeric features.

    Applies median imputation followed optionally by standard scaling.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaling:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
