"""
Data ingestion module for MLPipe.

Provides a structured Dataset abstraction and safe CSV loading with validation,
hashing, and descriptive error messages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mlpipe.core.exceptions import DatasetError
from mlpipe.utils.hashing import compute_file_hash


@dataclass
class Dataset:
    """Structured representation of an ingested dataset."""

    filepath: Path
    df: pd.DataFrame
    num_rows: int
    num_cols: int
    memory_bytes: int
    sha256_hash: str
    column_names: List[str]

    @property
    def filename(self) -> str:
        return self.filepath.name

    @property
    def memory_mb(self) -> float:
        return round(self.memory_bytes / (1024 * 1024), 2)

    def summary(self) -> Dict[str, Any]:
        """Return a clean dictionary summary of the dataset."""
        return {
            "filename": self.filename,
            "filepath": str(self.filepath),
            "rows": self.num_rows,
            "columns": self.num_cols,
            "memory_mb": self.memory_mb,
            "sha256": self.sha256_hash,
            "column_names": self.column_names,
        }


def load_dataset(path: Union[str, Path]) -> Dataset:
    """
    Safely load a CSV dataset from disk and return a structured Dataset object.

    Args:
        path: Path to the CSV file.

    Returns:
        Dataset object containing metadata and dataframe.

    Raises:
        DatasetError: If the file does not exist, has an invalid format, or is empty.
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        raise DatasetError(
            f"Dataset file not found at '{file_path}'.",
            "Verify that the file path is correct and accessible."
        )

    if not file_path.is_file():
        raise DatasetError(
            f"Path '{file_path}' is a directory, not a file.",
            "Provide the path to a valid CSV file."
        )

    if file_path.suffix.lower() not in [".csv"]:
        raise DatasetError(
            f"Unsupported file format '{file_path.suffix}'. MLPipe currently supports '.csv'.",
            "Please convert your dataset to CSV format."
        )

    # Check file size (e.g. empty file)
    if file_path.stat().st_size == 0:
        raise DatasetError(
            f"Dataset file '{file_path.name}' is completely empty (0 bytes).",
            "Ensure the file contains valid CSV data with a header and rows."
        )

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except pd.errors.EmptyDataError:
        raise DatasetError(
            f"Dataset file '{file_path.name}' contains no headers or data.",
            "Check that the CSV has column names and at least one data row."
        )
    except pd.errors.ParserError as e:
        raise DatasetError(
            f"Failed to parse CSV file '{file_path.name}': {e}",
            "Check for delimiter issues, unescaped quotes, or inconsistent row lengths."
        )
    except Exception as e:
        raise DatasetError(
            f"Unexpected error while reading '{file_path.name}': {e}",
            "Verify file permissions and that the file is not locked by another application."
        )

    if df.empty or len(df) == 0:
        raise DatasetError(
            f"Dataset '{file_path.name}' contains headers but zero data rows.",
            "Provide a dataset with at least a few sample rows."
        )

    if len(df.columns) == 0:
        raise DatasetError(
            f"Dataset '{file_path.name}' contains no columns.",
            "Check the delimiter of your CSV file."
        )

    # Compute SHA-256
    file_hash = compute_file_hash(file_path)
    memory_usage = int(df.memory_usage(deep=True).sum())

    return Dataset(
        filepath=file_path,
        df=df,
        num_rows=len(df),
        num_cols=len(df.columns),
        memory_bytes=memory_usage,
        sha256_hash=file_hash,
        column_names=list(df.columns),
    )
