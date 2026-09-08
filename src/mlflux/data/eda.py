"""
Automated Exploratory Data Analysis (EDA) module for MLFlux.

Calculates target distributions, feature correlations, missingness, and outlier metrics
to present users with key insights prior to training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EDAReport:
    """Summary of exploratory data analysis performed on a dataset."""

    target_column: str
    task_type: str
    num_rows: int
    num_cols: int
    target_distribution: Dict[str, Any]
    correlations: List[Dict[str, Any]] = field(default_factory=list)
    missing_summary: List[Dict[str, Any]] = field(default_factory=list)
    outlier_counts: Dict[str, int] = field(default_factory=dict)


def perform_eda(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> EDAReport:
    """
    Perform fast, automated exploratory data analysis focused on the target and features.

    Args:
        df: Raw or clean input DataFrame.
        target_column: Column being predicted.
        task_type: 'classification' or 'regression'.

    Returns:
        EDAReport container with structured EDA statistics.
    """
    clean_df = df.dropna(subset=[target_column]).copy()
    y = clean_df[target_column]
    X = clean_df.drop(columns=[target_column])

    # 1. Target Distribution
    target_dist: Dict[str, Any] = {}
    if task_type == "classification":
        val_counts = y.value_counts()
        total = len(y)
        classes = []
        for cls_val, cnt in val_counts.items():
            classes.append({
                "class": str(cls_val),
                "count": int(cnt),
                "percentage": round((cnt / total) * 100, 1),
            })
        target_dist["classes"] = classes
        target_dist["total_samples"] = total
        target_dist["unique_classes"] = len(classes)
        # Class balance ratio (min class / max class)
        if len(classes) > 1:
            ratio = round(classes[-1]["count"] / classes[0]["count"], 2)
            target_dist["balance_ratio"] = ratio
    else:
        num_y = pd.to_numeric(y, errors="coerce")
        target_dist["mean"] = round(float(num_y.mean()), 4) if not num_y.empty else 0.0
        target_dist["std"] = round(float(num_y.std()), 4) if not num_y.empty else 0.0
        target_dist["min"] = round(float(num_y.min()), 4) if not num_y.empty else 0.0
        target_dist["median"] = round(float(num_y.median()), 4) if not num_y.empty else 0.0
        target_dist["max"] = round(float(num_y.max()), 4) if not num_y.empty else 0.0
        skew = float(num_y.skew()) if len(num_y) > 2 else 0.0
        target_dist["skew"] = round(skew, 2)

    # 2. Numeric Correlations with Target
    correlations: List[Dict[str, Any]] = []
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        target_series = None
        if task_type == "regression":
            target_series = pd.to_numeric(y, errors="coerce")
        elif task_type == "classification" and y.nunique() == 2:
            # Encode binary classes to 0 and 1 for correlation calculation
            unique_vals = list(y.unique())
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            target_series = y.map(mapping)

        if target_series is not None:
            corr_series = X[numeric_cols].corrwith(target_series).dropna().abs().sort_values(ascending=False)
            for col_name, corr_val in corr_series.head(5).items():
                correlations.append({
                    "feature": col_name,
                    "correlation": round(float(corr_val), 3),
                })

    # 3. Missing Value Summary
    missing_summary: List[Dict[str, Any]] = []
    missing = df.isnull().sum()
    for col_name, count in missing[missing > 0].items():
        pct = round((count / len(df)) * 100, 1)
        missing_summary.append({
            "column": col_name,
            "missing_count": int(count),
            "missing_pct": pct,
        })

    # 4. Outlier Detection on Numeric Features (IQR method)
    outlier_counts: Dict[str, int] = {}
    for col in numeric_cols[:10]:
        series = pd.to_numeric(X[col], errors="coerce").dropna()
        if len(series) >= 10:
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                count = int(((series < lower) | (series > upper)).sum())
                if count > 0:
                    outlier_counts[col] = count

    return EDAReport(
        target_column=target_column,
        task_type=task_type,
        num_rows=len(df),
        num_cols=len(df.columns),
        target_distribution=target_dist,
        correlations=correlations,
        missing_summary=missing_summary,
        outlier_counts=outlier_counts,
    )
