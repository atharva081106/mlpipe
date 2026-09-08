"""
Data profiling module for MLFlux.

Calculates comprehensive statistics on datasets and columns without hardcoded values.
Detects column types, distributions, missingness, and structural issues.
"""

from dataclasses import asdict, dataclass, field
import json
import math
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mlflux.data.ingestion import Dataset


def _safe_val(val: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable standard Python types."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    return str(val)


def detect_column_type(series: pd.Series) -> str:
    """Infer column type: 'numeric', 'categorical', 'boolean', or 'datetime'."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        # Even if numeric, check if it's strictly binary 0/1 with boolean meaning
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1}) and len(unique_vals) <= 2:
            # Let boolean be inferred if column name or values suggest it
            name_lower = str(series.name).lower()
            if any(p in name_lower for p in ["is_", "has_", "flag", "active", "churn"]):
                return "categorical"  # target or binary categorical
        return "numeric"

    # For object/string columns, try datetime detection
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            sample = non_null.head(min(50, len(non_null)))
            try:
                pd.to_datetime(sample, format="mixed")
                # Also check full or larger sample if small sample succeeded
                return "datetime"
            except Exception:
                pass

            # Check boolean-like strings
            lower_sample = set(str(v).strip().lower() for v in sample.unique())
            if lower_sample.issubset({"true", "false", "yes", "no", "t", "f", "1", "0", "y", "n"}):
                return "boolean"

    return "categorical"


@dataclass
class ColumnProfile:
    """Profile of an individual dataset column."""

    name: str
    detected_type: str
    missing_count: int
    missing_pct: float
    unique_count: int
    examples: List[Any]
    # Numeric stats
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    # Categorical stats
    top_values: List[Dict[str, Any]] = field(default_factory=list)
    # Datetime stats
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    # Warnings/Flags
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetProfile:
    """Complete dataset profile including dataset-level and column-level information."""

    num_rows: int
    num_cols: int
    memory_bytes: int
    memory_mb: float
    duplicate_rows: int
    total_missing_values: int
    columns: List[ColumnProfile]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        for col in self.columns:
            if col.name == name:
                return col
        return None


def profile_dataset(dataset: Union[Dataset, pd.DataFrame]) -> DatasetProfile:
    """
    Profile a dataset and generate accurate, computed descriptive statistics.

    Args:
        dataset: Ingested Dataset or pandas DataFrame.

    Returns:
        DatasetProfile object.
    """
    df = dataset.df if isinstance(dataset, Dataset) else dataset

    num_rows = len(df)
    num_cols = len(df.columns)
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_mb = round(memory_bytes / (1024 * 1024), 2)
    duplicate_rows = int(df.duplicated().sum())
    total_missing = int(df.isna().sum().sum())

    column_profiles: List[ColumnProfile] = []
    dataset_warnings: List[str] = []

    if duplicate_rows > 0:
        dataset_warnings.append(f"Found {duplicate_rows} duplicate rows ({round(duplicate_rows/num_rows*100, 1)}%).")

    id_regex = re.compile(r"(^id$|_id$|^id_|^index$|^guid$|^uuid$)", re.IGNORECASE)

    for col_name in df.columns:
        series = df[col_name]
        col_type = detect_column_type(series)
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / num_rows) * 100, 2) if num_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        examples = [_safe_val(x) for x in series.dropna().unique()[:5].tolist()]
        flags = []

        # Check flags
        if missing_pct > 50.0:
            flags.append(f"High missingness: {missing_pct}% missing")
        if unique_count <= 1:
            flags.append("Constant column (only 1 unique value)")
        elif unique_count == num_rows and col_type in ("numeric", "categorical"):
            flags.append("Suspicious unique/ID column (every value is distinct)")
        elif id_regex.search(str(col_name)) and unique_count > num_rows * 0.8:
            flags.append("Possible identifier/index column")

        if col_type == "categorical" and unique_count > 100 and unique_count > num_rows * 0.5:
            flags.append(f"Extremely high cardinality: {unique_count} distinct categories")

        min_val = max_val = mean_val = median_val = std_val = None
        top_values: List[Dict[str, Any]] = []
        min_date = max_date = None

        if col_type == "numeric":
            desc = series.describe()
            min_val = _safe_val(desc.get("min"))
            max_val = _safe_val(desc.get("max"))
            mean_val = _safe_val(desc.get("mean"))
            median_val = _safe_val(series.median())
            std_val = _safe_val(desc.get("std"))
        elif col_type == "categorical":
            vc = series.value_counts(dropna=True).head(5)
            top_values = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
        elif col_type == "datetime":
            try:
                dt_series = pd.to_datetime(series, errors="coerce")
                min_date = _safe_val(dt_series.min())
                max_date = _safe_val(dt_series.max())
            except Exception:
                pass

        col_prof = ColumnProfile(
            name=str(col_name),
            detected_type=col_type,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            examples=examples,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            median_val=median_val,
            std_val=std_val,
            top_values=top_values,
            min_date=min_date,
            max_date=max_date,
            flags=flags,
        )
        column_profiles.append(col_prof)

    return DatasetProfile(
        num_rows=num_rows,
        num_cols=num_cols,
        memory_bytes=memory_bytes,
        memory_mb=memory_mb,
        duplicate_rows=duplicate_rows,
        total_missing_values=total_missing,
        columns=column_profiles,
        warnings=dataset_warnings,
    )
