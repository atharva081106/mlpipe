"""Intelligent recommendation engine for MLPipe CLI.

Provides data-driven recommendations and rationale for:
1. Target column selection
2. Train / test split ratios
3. Candidate model selection
4. Hyperparameter fine-tuning options
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


COMMON_TARGET_KEYWORDS = [
    "target", "label", "class", "outcome", "churn", "status", "price",
    "sales", "survived", "y", "revenue", "salary", "default", "rating",
    "score", "risk", "attrition", "fraud", "deposit", "purchased", "bought",
    "response", "diagnosis", "quality", "charges", "median_house_value",
    "tip", "fare", "delay", "approved", "admitted", "converted"
]

NON_TARGET_KEYWORDS = [
    "id", "uuid", "guid", "index", "unnamed", "row", "serial", "key", "created_at", "updated_at", "timestamp"
]


def recommend_target_column(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Intelligently inspects the dataframe and recommends the most likely target column
    along with an explanation of why.
    """
    cols = list(df.columns)
    if not cols:
        raise ValueError("DataFrame has no columns")

    scores: Dict[str, float] = {}
    reasons: Dict[str, str] = {}

    for idx, col in enumerate(cols):
        col_lower = str(col).lower().strip()
        score = 0.0
        reason_parts = []

        # Penalize obvious ID columns
        is_id = any(re.search(rf"\b{k}\b", col_lower) for k in NON_TARGET_KEYWORDS) or col_lower in ["id", "index", "unnamed: 0"]
        if is_id:
            score -= 50.0

        # Exact match with standard target names
        if col_lower in COMMON_TARGET_KEYWORDS:
            score += 40.0
            reason_parts.append(f"Standard target name '{col}'")
        elif any(k in col_lower for k in COMMON_TARGET_KEYWORDS):
            score += 25.0
            reason_parts.append("Contains target keyword")

        # Position heuristic: Last column is conventional ML target
        if idx == len(cols) - 1:
            score += 15.0
            reason_parts.append("Conventional position (last column in dataset)")
        elif idx == 0 and not is_id:
            score += 2.0

        # Target feasibility heuristics
        series = df[col].dropna()
        n_unique = series.nunique()
        n_total = len(series)

        if n_total > 0:
            # Check for binary classification target
            if n_unique == 2:
                score += 20.0
                reason_parts.append("Binary outcome (ideal for classification)")
            # Check for multiclass target (3 to 15 unique values and non-float or integer)
            elif 3 <= n_unique <= 15 and (series.dtype == "object" or series.dtype.name == "category" or pd.api.types.is_integer_dtype(series)):
                score += 12.0
                reason_parts.append(f"Categorical outcome with {n_unique} classes")
            # Check for continuous target (float or wide-range numeric)
            elif pd.api.types.is_numeric_dtype(series) and n_unique > 20 and not is_id:
                score += 10.0
                reason_parts.append("Continuous numerical target (ideal for regression)")

            # Check missingness: heavy missingness is bad for a target
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > 0.3:
                score -= 30.0

        scores[col] = score
        reasons[col] = "; ".join(reason_parts) if reason_parts else "Candidate column"

    # Pick the highest scoring column
    best_col = max(cols, key=lambda c: scores[c])
    best_reason = reasons[best_col]

    return best_col, best_reason


def recommend_split_strategy(num_rows: int, task_type: str) -> Dict[str, Any]:
    """
    Recommends train/test split ratio and strategy based on dataset size.
    """
    if num_rows < 100:
        ratio = 0.25
        explanation = "75% Train / 25% Test (Recommended for small datasets to retain sufficient test evaluation samples)"
    elif num_rows > 100_000:
        ratio = 0.10
        explanation = "90% Train / 10% Test (Recommended for large datasets where 10% yields >10,000 test rows)"
    else:
        ratio = 0.20
        explanation = "80% Train / 20% Test (Industry standard balanced split with zero data leakage)"

    stratified = task_type == "classification"
    strat_text = "Stratified (preserves class label distribution in both sets)" if stratified else "Random shuffle (preserves continuous distribution)"

    return {
        "test_size": ratio,
        "explanation": explanation,
        "is_stratified": stratified,
        "stratification_note": strat_text,
    }


def recommend_models_for_task(
    task_type: str,
    num_rows: int,
    candidate_names: List[str],
) -> Dict[str, Any]:
    """
    Provides data-informed recommendations for candidate models.
    """
    model_notes: Dict[str, str] = {}
    for name in candidate_names:
        nl = name.lower()
        if "histgradient" in nl:
            model_notes[name] = "⭐ Highly Recommended — State-of-the-art LightGBM-style trees, exceptional accuracy and fast on tabular data."
        elif "randomforest" in nl or "random forest" in nl:
            model_notes[name] = "⭐ Recommended — Robust non-linear ensemble, low overfitting risk, excellent generalizer."
        elif "extratrees" in nl or "extra trees" in nl:
            model_notes[name] = "Fast randomized trees, provides high variance reduction on noisy tabular features."
        elif "logistic" in nl:
            model_notes[name] = "Linear baseline — fast, highly interpretable probabilistic model."
        elif "ridge" in nl:
            model_notes[name] = "Linear baseline — L2 regularized regression resistant to multicollinearity."
        else:
            model_notes[name] = "Standard candidate model."

    all_recommendation = (
        "Option [A] (All Models) is the strongly recommended best practice — trains and compares "
        "all candidates using 5-fold Cross-Validation to empirically discover the top performer for your dataset."
    )

    return {
        "candidate_notes": model_notes,
        "recommended_choice": "A",
        "all_recommendation": all_recommendation,
    }


def recommend_hyperparameter_overrides(
    model_name: str,
    current_params: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Suggests sensible fine-tuning values for key parameters with rationale.
    """
    mn = model_name.lower()
    recommendations: Dict[str, Dict[str, Any]] = {}

    if "random forest" in mn or "randomforest" in mn or "extra trees" in mn or "extratrees" in mn:
        if "n_estimators" in current_params:
            cur = current_params["n_estimators"]
            rec = min(300, max(50, int(cur * 1.5))) if isinstance(cur, (int, float)) else 150
            recommendations["n_estimators"] = {
                "current": cur,
                "recommended": rec,
                "rationale": "More trees lower model variance and stabilize ensemble predictions.",
            }
        if "max_depth" in current_params:
            cur = current_params["max_depth"]
            rec = 12 if cur is None else (cur + 4)
            recommendations["max_depth"] = {
                "current": cur,
                "recommended": rec,
                "rationale": "Constraining tree depth prevents overfitting on noisy leaf splits.",
            }
        if "min_samples_split" in current_params:
            cur = current_params["min_samples_split"]
            rec = 5 if cur == 2 else 2
            recommendations["min_samples_split"] = {
                "current": cur,
                "recommended": rec,
                "rationale": "Higher minimum split requires more evidence before splitting a node.",
            }

    elif "histgradient" in mn:
        if "learning_rate" in current_params:
            cur = current_params["learning_rate"]
            rec = 0.05 if cur >= 0.1 else 0.1
            recommendations["learning_rate"] = {
                "current": cur,
                "recommended": rec,
                "rationale": "A slightly lower learning rate with more iterations improves generalization.",
            }
        if "max_iter" in current_params:
            cur = current_params.get("max_iter", 100)
            rec = 150
            recommendations["max_iter"] = {
                "current": cur,
                "recommended": rec,
                "rationale": "Allows gradient boosting more sequential boosting steps to minimize loss.",
            }
        if "l2_regularization" in current_params:
            cur = current_params["l2_regularization"]
            rec = 0.1 if cur == 0.0 else 0.0
            recommendations["l2_regularization"] = {
                "current": cur,
                "recommended": rec,
                "rationale": "L2 leaf penalty penalizes complex trees and combats overfitting.",
            }

    elif "logistic" in mn or "ridge" in mn:
        if "C" in current_params:
            cur = current_params["C"]
            rec = 0.1 if cur >= 1.0 else 1.0
            recommendations["C"] = {
                "current": cur,
                "recommended": rec,
                "rationale": "Smaller C implies stronger L2 regularization, preventing coefficient explosion.",
            }
        if "alpha" in current_params:
            cur = current_params["alpha"]
            rec = 10.0 if cur <= 1.0 else 1.0
            recommendations["alpha"] = {
                "current": cur,
                "recommended": rec,
                "rationale": "Higher alpha adds stronger penalty to shrink colinear regression weights.",
            }

    return recommendations
