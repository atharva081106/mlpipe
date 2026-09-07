"""
Core Pipeline orchestrator for MLPipe.

Provides the primary user-facing Python API for automated tabular machine learning:
data ingestion, validation, splitting, preprocessing, multi-model training,
cross-validation, hyperparameter tuning, model evaluation, explainability,
and artifact management.
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline

from mlpipe.artifacts.manager import ArtifactManager, load_pipeline_artifact
from mlpipe.artifacts.serialization import save_joblib
from mlpipe.core.config import PipelineConfig, TaskType, TrainingMode
from mlpipe.core.exceptions import (
    ConfigurationError,
    DatasetError,
    PipelineError,
    PredictionError,
    TrainingError,
    ValidationError,
)
from mlpipe.core.result import PipelineResult
from mlpipe.data.ingestion import Dataset, load_dataset
from mlpipe.data.profiling import DatasetProfile, profile_dataset
from mlpipe.data.splitting import SplitData, split_data
from mlpipe.data.validation import ValidationReport, validate_dataset
from mlpipe.evaluation.evaluator import build_leaderboard, evaluate_pipeline_on_test
from mlpipe.evaluation.metrics import select_primary_metric
from mlpipe.explainability.importance import extract_feature_importance
from mlpipe.models.selection import get_candidates_for_task
from mlpipe.preprocessing.builder import classify_columns, build_preprocessor
from mlpipe.tuning.search import TuningResult, tune_candidate
from mlpipe.utils.logging import configure_logging, get_logger

logger = get_logger("pipeline")


class Pipeline:
    """
    MLPipe Automated Machine Learning Pipeline.

    Example:
    ```python
    from mlpipe import Pipeline

    pipe = Pipeline(target="churn", task="auto", mode="balanced")
    result = pipe.fit("customer_churn.csv")

    print(result.best_model)
    print(result.metrics)

    predictions = pipe.predict("new_data.csv")
    pipe.save("./model")
    ```
    """

    def __init__(
        self,
        target: str,
        task: Union[TaskType, str] = TaskType.AUTO,
        mode: Union[TrainingMode, str] = TrainingMode.BALANCED,
        test_size: float = 0.20,
        random_seed: int = 42,
        cv_folds: int = 5,
        output_dir: Union[str, Path] = "./mlpipe_runs",
        verbose: bool = False,
        selected_models: Optional[List[str]] = None,
    ):
        self.config = PipelineConfig(
            target=target,
            task=task,
            mode=mode,
            test_size=test_size,
            random_seed=random_seed,
            cv_folds=cv_folds,
            output_dir=Path(output_dir),
            verbose=verbose,
            selected_models=selected_models,
        )
        configure_logging(verbose=verbose)

        self._fitted_pipeline: Optional[SklearnPipeline] = None
        self._result: Optional[PipelineResult] = None
        self._profile: Optional[DatasetProfile] = None
        self._validation: Optional[ValidationReport] = None
        self._split_res: Optional[SplitData] = None

    @property
    def is_fitted(self) -> bool:
        """Whether the pipeline has been fitted."""
        return self._fitted_pipeline is not None

    @property
    def result(self) -> Optional[PipelineResult]:
        """Outcome of the latest fit operation."""
        return self._result

    @property
    def profile(self) -> Optional[DatasetProfile]:
        """Dataset profile calculated during fit."""
        return self._profile

    @property
    def validation(self) -> Optional[ValidationReport]:
        """Validation report generated during fit."""
        return self._validation

    def fit(
        self,
        data: Union[str, Path, pd.DataFrame, Dataset],
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> PipelineResult:
        """
        Execute the complete automated machine learning lifecycle on the input dataset.

        Args:
            data: Filepath to CSV, pandas DataFrame, or MLPipe Dataset.
            on_progress: Optional callback invoked with (stage, message).

        Returns:
            PipelineResult with metrics, leaderboard, and artifact locations.
        """
        def _notify(stage: str, msg: str):
            logger.info("[%s] %s", stage, msg)
            if on_progress:
                on_progress(stage, msg)

        t_start = time.perf_counter()

        # ── 1. Data Ingestion ──────────────────────────────────────────────
        _notify("ingestion", "Loading dataset...")
        if isinstance(data, (str, Path)):
            dataset = load_dataset(data)
        elif isinstance(data, Dataset):
            dataset = data
        elif isinstance(data, pd.DataFrame):
            dataset = Dataset(
                filepath=Path("in_memory.csv"),
                df=data.copy(),
                num_rows=len(data),
                num_cols=len(data.columns),
                memory_bytes=int(data.memory_usage(deep=True).sum()),
                sha256_hash="in_memory_dataframe",
                column_names=list(data.columns),
            )
        else:
            raise DatasetError(f"Unsupported data type '{type(data)}'. Provide a CSV path or DataFrame.")

        # ── 2. Data Profiling ──────────────────────────────────────────────
        _notify("profiling", "Profiling dataset structure and distributions...")
        self._profile = profile_dataset(dataset)

        # ── 3. Data Validation ─────────────────────────────────────────────
        _notify("validation", "Validating dataset and task compatibility...")
        self._validation = validate_dataset(
            dataset=dataset,
            target_column=self.config.target,
            task_override=self.config.task.value,
            test_size=self.config.test_size,
            cv_folds=self.config.cv_folds,
        )
        self._validation.raise_if_invalid()

        task_type = self._validation.detected_task
        _notify("task_detection", f"Task detected: {task_type.capitalize()}")

        # ── 4. Train / Test Splitting ──────────────────────────────────────
        _notify("splitting", f"Splitting data ({int((1-self.config.test_size)*100)}% train / {int(self.config.test_size*100)}% test)...")
        split_res: SplitData = split_data(
            df=dataset.df,
            target_column=self.config.target,
            task_type=task_type,
            test_size=self.config.test_size,
            random_seed=self.config.random_seed,
        )
        self._split_res = split_res

        # ── 5. Preprocessing Pipeline Construction ─────────────────────────
        _notify("preprocessing", "Building feature preprocessing pipeline...")
        # Classify columns strictly from training features to avoid leakage
        column_assignments = classify_columns(split_res.X_train)
        preprocessor = build_preprocessor(column_assignments, with_scaling=True)

        # ── 6. Metric Selection ────────────────────────────────────────────
        primary_metric = select_primary_metric(task_type, split_res.y_train)
        _notify("metric_selection", f"Primary evaluation metric: {primary_metric}")

        # ── 7. Model Selection & Tuning ────────────────────────────────────
        candidates = get_candidates_for_task(task_type, mode=self.config.mode.value)
        if self.config.selected_models:
            sel_norm = [m.lower().strip() for m in self.config.selected_models]
            filtered = [
                c for c in candidates
                if any(s in c.name.lower() or c.name.lower() in s for s in sel_norm)
            ]
            if filtered:
                candidates = filtered

        tuning_results: List[TuningResult] = []

        for candidate in candidates:
            _notify("training", f"Training & tuning candidate: {candidate.name}...")
            res = tune_candidate(
                candidate=candidate,
                preprocessor=preprocessor,
                X_train=split_res.X_train,
                y_train=split_res.y_train,
                primary_metric=primary_metric,
                cv_folds=self.config.cv_folds,
                mode=self.config.mode.value,
                random_seed=self.config.random_seed,
            )
            tuning_results.append(res)

        # Check if all models failed
        successful_runs = [r for r in tuning_results if r.best_pipeline is not None and not r.error]
        if not successful_runs:
            error_details = "\n".join(f"- {r.candidate_name}: {r.error}" for r in tuning_results)
            raise TrainingError(
                f"All candidate models failed during training:\n{error_details}",
                "Review the errors above and ensure features are compatible."
            )

        # ── 8. Leaderboard & Model Selection ───────────────────────────────
        _notify("evaluation", "Compiling leaderboard and selecting winning model...")
        leaderboard = build_leaderboard(
            tuning_results=tuning_results,
            X_test=split_res.X_test,
            y_test=split_res.y_test,
            task_type=task_type,
            primary_metric=primary_metric,
        )

        winning_entry = leaderboard[0]
        best_model_name = winning_entry["model"]
        best_cv_score = winning_entry["cv_score"]
        test_score = winning_entry["test_score"]
        test_metrics = winning_entry.get("test_metrics", {})

        # Find winning pipeline
        winning_result = next(r for r in tuning_results if r.candidate_name == best_model_name)
        best_pipeline = winning_result.best_pipeline
        self._fitted_pipeline = best_pipeline

        # ── 9. Explainability ──────────────────────────────────────────────
        _notify("explainability", "Extracting feature importances...")
        feature_importance = extract_feature_importance(best_pipeline)

        # ── 10. Test Set Predictions & Preview ─────────────────────────────
        _notify("evaluation", "Generating hold-out test set predictions...")
        y_test_pred = best_pipeline.predict(split_res.X_test)

        test_predictions_df = split_res.X_test.copy()
        test_predictions_df[f"actual_{self.config.target}"] = split_res.y_test.values
        test_predictions_df[f"predicted_{self.config.target}"] = y_test_pred

        test_preview = []
        preview_n = min(10, len(split_res.X_test))
        candidate_cols = [c for c in split_res.X_test.columns if "id" not in c.lower() and "uuid" not in c.lower()]
        if not candidate_cols:
            candidate_cols = list(split_res.X_test.columns)
        preview_cols = candidate_cols[:3]

        if task_type == "classification":
            matches = (split_res.y_test.values == y_test_pred)
            test_predictions_df["match"] = matches
            for i in range(preview_n):
                sample_feats = {
                    col: (round(float(split_res.X_test.iloc[i][col]), 2) if isinstance(split_res.X_test.iloc[i][col], (float, np.floating)) else split_res.X_test.iloc[i][col])
                    for col in preview_cols
                }
                test_preview.append({
                    "row_idx": int(i + 1),
                    "actual": split_res.y_test.values[i],
                    "predicted": y_test_pred[i],
                    "match": bool(matches[i]),
                    "features": sample_feats,
                })
        else:
            abs_err = np.abs(split_res.y_test.values - y_test_pred)
            test_predictions_df["absolute_error"] = np.round(abs_err, 4)
            for i in range(preview_n):
                sample_feats = {
                    col: (round(float(split_res.X_test.iloc[i][col]), 2) if isinstance(split_res.X_test.iloc[i][col], (float, np.floating)) else split_res.X_test.iloc[i][col])
                    for col in preview_cols
                }
                test_preview.append({
                    "row_idx": int(i + 1),
                    "actual": float(np.round(split_res.y_test.values[i], 4)),
                    "predicted": float(np.round(y_test_pred[i], 4)),
                    "error": float(np.round(abs_err[i], 4)),
                    "features": sample_feats,
                })

        # ── 11. Artifact Generation ────────────────────────────────────────
        _notify("artifacts", "Saving run artifacts and pipeline...")
        artifact_mgr = ArtifactManager(base_output_dir=self.config.output_dir)
        run_id = artifact_mgr.generate_run_id()
        elapsed_s = round(time.perf_counter() - t_start, 2)

        saved_dir = artifact_mgr.save_run_artifacts(
            run_id=run_id,
            pipeline=best_pipeline,
            dataset_summary=dataset.summary(),
            target_column=self.config.target,
            task_type=task_type,
            training_mode=self.config.mode.value,
            primary_metric=primary_metric,
            best_model_name=best_model_name,
            best_cv_score=best_cv_score,
            test_score=test_score,
            test_metrics=test_metrics,
            leaderboard=leaderboard,
            feature_importance=feature_importance,
            elapsed_time_s=elapsed_s,
            random_seed=self.config.random_seed,
            train_df=split_res.train_df,
            test_df=split_res.test_df,
            test_predictions_df=test_predictions_df,
        )

        self._result = PipelineResult(
            run_id=run_id,
            target_column=self.config.target,
            task_type=task_type,
            best_model_name=best_model_name,
            primary_metric=primary_metric,
            best_cv_score=best_cv_score,
            test_score=test_score,
            leaderboard=leaderboard,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            artifacts_dir=saved_dir,
            elapsed_time_s=elapsed_s,
            metadata={
                "training_mode": self.config.mode.value,
                "random_seed": self.config.random_seed,
                "train_rows": split_res.train_size,
                "test_rows": split_res.test_size,
            },
            test_preview=test_preview,
            train_path=saved_dir / "train.csv",
            test_path=saved_dir / "test.csv",
            test_predictions_path=saved_dir / "test_predictions.csv",
        )

        _notify("complete", f"Pipeline complete in {elapsed_s}s. Best model: {best_model_name} ({best_cv_score})")
        return self._result

    def predict(self, data: Union[str, Path, pd.DataFrame]) -> np.ndarray:
        """
        Generate predictions using the fitted pipeline.

        Args:
            data: Path to CSV or pandas DataFrame.

        Returns:
            Numpy array of predictions.
        """
        if not self.is_fitted or self._fitted_pipeline is None:
            raise PredictionError(
                "Pipeline has not been fitted or loaded.",
                "Call pipeline.fit(data) or Pipeline.load(path) before calling predict."
            )

        if isinstance(data, (str, Path)):
            dataset = load_dataset(data)
            df = dataset.df
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise PredictionError(f"Unsupported data type '{type(data)}' for prediction.")

        # Drop target column if present in prediction data (e.g. test set)
        if self.config.target in df.columns:
            df = df.drop(columns=[self.config.target])

        try:
            return self._fitted_pipeline.predict(df)
        except Exception as e:
            raise PredictionError(
                f"Prediction failed: {e}",
                "Ensure that the input data contains all required feature columns with compatible formats."
            )

    def predict_proba(self, data: Union[str, Path, pd.DataFrame]) -> np.ndarray:
        """
        Generate prediction probabilities for classification tasks.
        """
        if not self.is_fitted or self._fitted_pipeline is None:
            raise PredictionError("Pipeline has not been fitted or loaded.")

        if not hasattr(self._fitted_pipeline, "predict_proba"):
            raise PredictionError("The underlying best model does not support predict_proba.")

        if isinstance(data, (str, Path)):
            dataset = load_dataset(data)
            df = dataset.df
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise PredictionError(f"Unsupported data type '{type(data)}' for prediction.")

        if self.config.target in df.columns:
            df = df.drop(columns=[self.config.target])

        return self._fitted_pipeline.predict_proba(df)

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save the fitted pipeline to a file or directory.

        Args:
            path: Destination file or directory.
        """
        if not self.is_fitted or self._fitted_pipeline is None:
            raise PipelineError("Cannot save an unfitted pipeline.")

        dest = Path(path).resolve()
        if dest.suffix == ".joblib":
            return save_joblib(self._fitted_pipeline, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            return save_joblib(self._fitted_pipeline, dest / "pipeline.joblib")

    @classmethod
    def load(cls, path: Union[str, Path], target: str = "unknown") -> "Pipeline":
        """
        Load a previously trained pipeline from disk.

        Args:
            path: Directory containing 'pipeline.joblib' or path to a .joblib file.
            target: Optional target name for consistency.

        Returns:
            Configured Pipeline instance ready for prediction.
        """
        fitted_pipe = load_pipeline_artifact(path)
        instance = cls(target=target)
        instance._fitted_pipeline = fitted_pipe
        return instance

    def retune_best_model(self, param_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retune the winning model with specific parameter overrides and evaluate on the test set.

        Args:
            param_overrides: Dictionary of hyperparameter overrides (e.g. {'n_estimators': 200}).

        Returns:
            Dictionary with old and new test scores, parameters, and updated metrics.
        """
        if not self.is_fitted or self._fitted_pipeline is None or self._result is None:
            raise PipelineError("Pipeline must be fitted before fine-tuning.")
        if self._split_res is None:
            raise PipelineError("No training split available for fine-tuning.")

        estimator = self._fitted_pipeline.named_steps.get("estimator")
        if estimator is None:
            raise PipelineError("Pipeline does not contain an estimator step.")

        est_class = estimator.__class__
        current_params = estimator.get_params()

        # Update params
        updated_params = dict(current_params)
        for k, v in param_overrides.items():
            if k in current_params:
                updated_params[k] = v

        # Build clean kwargs
        valid_kwargs = {k: v for k, v in updated_params.items() if not k.startswith("_")}
        new_estimator = est_class(**valid_kwargs)

        preprocessor = self._fitted_pipeline.named_steps.get("preprocessor")
        new_pipeline = SklearnPipeline([
            ("preprocessor", preprocessor),
            ("estimator", new_estimator),
        ])

        # Fit on training split
        new_pipeline.fit(self._split_res.X_train, self._split_res.y_train)

        # Evaluate on hold-out test split
        test_metrics = evaluate_pipeline_on_test(
            pipeline=new_pipeline,
            X_test=self._split_res.X_test,
            y_test=self._split_res.y_test,
            task_type=self._result.task_type,
        )
        new_test_score = test_metrics.get(self._result.primary_metric, 0.0)

        # Update active fitted pipeline
        self._fitted_pipeline = new_pipeline

        return {
            "model": self._result.best_model_name,
            "old_test_score": self._result.test_score,
            "new_test_score": new_test_score,
            "primary_metric": self._result.primary_metric,
            "params": param_overrides,
            "test_metrics": test_metrics,
        }
