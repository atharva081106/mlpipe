"""Data ingestion, profiling, validation, and splitting for MLPipe."""

from mlpipe.data.ingestion import Dataset, load_dataset
from mlpipe.data.profiling import ColumnProfile, DatasetProfile, detect_column_type, profile_dataset
from mlpipe.data.splitting import SplitData, split_data
from mlpipe.data.validation import ValidationReport, detect_task, validate_dataset

__all__ = [
    "Dataset",
    "load_dataset",
    "ColumnProfile",
    "DatasetProfile",
    "detect_column_type",
    "profile_dataset",
    "SplitData",
    "split_data",
    "ValidationReport",
    "detect_task",
    "validate_dataset",
]
