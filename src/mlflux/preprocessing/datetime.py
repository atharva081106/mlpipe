"""Datetime feature extraction transformer."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts tabular calendar features (year, month, day, dayofweek) from datetime columns.
    """

    def __init__(self):
        self.feature_names_: List[str] = []

    def fit(self, X, y=None):
        cols = X.columns if hasattr(X, "columns") else [f"dt_{i}" for i in range(X.shape[1])]
        names = []
        for col in cols:
            names.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek",
            ])
        self.feature_names_ = names
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        extracted = []

        for col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            extracted.append(s.dt.year.fillna(-1).astype(float).values)
            extracted.append(s.dt.month.fillna(-1).astype(float).values)
            extracted.append(s.dt.day.fillna(-1).astype(float).values)
            extracted.append(s.dt.dayofweek.fillna(-1).astype(float).values)

        if not extracted:
            return np.empty((len(df), 0))

        return np.column_stack(extracted)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if self.feature_names_:
            return self.feature_names_
        if input_features is not None:
            names = []
            for col in input_features:
                names.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])
            return names
        return []
