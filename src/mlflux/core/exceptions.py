"""
Custom exceptions for MLFlux.

Provides domain-specific errors with user-friendly descriptions and suggested actions.
"""

from typing import Optional


class MLFluxError(Exception):
    """Base exception for all MLFlux errors."""

    def __init__(self, message: str, suggested_action: Optional[str] = None):
        self.message = message
        self.suggested_action = suggested_action
        full_msg = message
        if suggested_action:
            full_msg += f"\n\nSuggested action:\n{suggested_action}"
        super().__init__(full_msg)


class DatasetError(MLFluxError):
    """Raised when there is an issue loading, reading, or parsing a dataset."""
    pass


class ValidationError(MLFluxError):
    """Raised when dataset pre-training validation fails fatal checks."""
    pass


class PreprocessingError(MLFluxError):
    """Raised when feature preprocessing or transformation fails."""
    pass


class TrainingError(MLFluxError):
    """Raised when model training or hyperparameter tuning fails."""
    pass


class EvaluationError(MLFluxError):
    """Raised when model evaluation fails."""
    pass


class ArtifactError(MLFluxError):
    """Raised when saving, loading, or managing artifacts fails."""
    pass


class ConfigurationError(MLFluxError):
    """Raised when invalid options or configurations are provided."""
    pass


class PredictionError(MLFluxError):
    """Raised when generating predictions fails."""
    pass


class PipelineError(MLFluxError):
    """Raised when general pipeline execution fails."""
    pass

