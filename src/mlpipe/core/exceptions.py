"""
Custom exceptions for MLPipe.

Provides domain-specific errors with user-friendly descriptions and suggested actions.
"""

from typing import Optional


class MLPipeError(Exception):
    """Base exception for all MLPipe errors."""

    def __init__(self, message: str, suggested_action: Optional[str] = None):
        self.message = message
        self.suggested_action = suggested_action
        full_msg = message
        if suggested_action:
            full_msg += f"\n\nSuggested action:\n{suggested_action}"
        super().__init__(full_msg)


class DatasetError(MLPipeError):
    """Raised when there is an issue loading, reading, or parsing a dataset."""
    pass


class ValidationError(MLPipeError):
    """Raised when dataset pre-training validation fails fatal checks."""
    pass


class PreprocessingError(MLPipeError):
    """Raised when feature preprocessing or transformation fails."""
    pass


class TrainingError(MLPipeError):
    """Raised when model training or hyperparameter tuning fails."""
    pass


class EvaluationError(MLPipeError):
    """Raised when model evaluation fails."""
    pass


class ArtifactError(MLPipeError):
    """Raised when saving, loading, or managing artifacts fails."""
    pass


class ConfigurationError(MLPipeError):
    """Raised when invalid options or configurations are provided."""
    pass


class PredictionError(MLPipeError):
    """Raised when generating predictions fails."""
    pass


class PipelineError(MLPipeError):
    """Raised when general pipeline execution fails."""
    pass

