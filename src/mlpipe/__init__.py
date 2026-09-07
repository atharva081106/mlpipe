"""
MLPipe: Automated Machine Learning Library and Terminal CLI.
"""

from mlpipe.core.config import PipelineConfig, TaskType, TrainingMode
from mlpipe.core.exceptions import (
    ArtifactError,
    ConfigurationError,
    DatasetError,
    EvaluationError,
    MLPipeError,
    PipelineError,
    PredictionError,
    PreprocessingError,
    TrainingError,
    ValidationError,
)
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.result import PipelineResult
from mlpipe.data.profiling import DatasetProfile
from mlpipe.data.validation import ValidationReport
from mlpipe.version import __version__

__all__ = [
    "Pipeline",
    "PipelineResult",
    "DatasetProfile",
    "ValidationReport",
    "PipelineConfig",
    "TaskType",
    "TrainingMode",
    "MLPipeError",
    "DatasetError",
    "ValidationError",
    "PreprocessingError",
    "TrainingError",
    "EvaluationError",
    "ArtifactError",
    "ConfigurationError",
    "PredictionError",
    "PipelineError",
    "__version__",
]
