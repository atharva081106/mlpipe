"""
Configuration models and defaults for MLPipe.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from mlpipe.core.exceptions import ConfigurationError


class TaskType(str, Enum):
    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TrainingMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


@dataclass
class PipelineConfig:
    """Configuration settings for an MLPipe run."""

    target: str
    task: Union[TaskType, str] = TaskType.AUTO
    mode: Union[TrainingMode, str] = TrainingMode.BALANCED
    test_size: float = 0.20
    random_seed: int = 42
    cv_folds: int = 5
    output_dir: Path = field(default_factory=lambda: Path("./mlpipe_runs"))
    verbose: bool = False

    def __post_init__(self):
        # Normalize and validate task
        if isinstance(self.task, str):
            try:
                self.task = TaskType(self.task.lower())
            except ValueError:
                valid = [t.value for t in TaskType]
                raise ConfigurationError(
                    f"Invalid task '{self.task}'. Must be one of {valid}.",
                    "Use 'auto', 'classification', or 'regression'."
                )

        # Normalize and validate mode
        if isinstance(self.mode, str):
            try:
                self.mode = TrainingMode(self.mode.lower())
            except ValueError:
                valid = [m.value for m in TrainingMode]
                raise ConfigurationError(
                    f"Invalid training mode '{self.mode}'. Must be one of {valid}.",
                    "Use 'fast', 'balanced', or 'thorough'."
                )

        # Validate test_size
        if not (0.05 <= self.test_size <= 0.5):
            raise ConfigurationError(
                f"Invalid test_size '{self.test_size}'. Must be between 0.05 and 0.50.",
                "Choose a reasonable test split ratio, such as 0.20 (20%)."
            )

        # Validate cv_folds
        if self.cv_folds < 2:
            raise ConfigurationError(
                f"cv_folds must be at least 2, got {self.cv_folds}.",
                "Set cv_folds to at least 2, typically 3 or 5."
            )

        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
