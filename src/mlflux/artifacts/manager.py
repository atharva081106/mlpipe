"""
Artifact management and inspection for MLFlux runs.

Creates reproducible artifact bundles containing model, pipeline, metrics,
metadata, and textual summaries.
"""

from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Union
import uuid

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mlflux.artifacts.serialization import load_joblib, load_json, save_joblib, save_json
from mlflux.core.exceptions import ArtifactError
from mlflux.version import __version__


class ArtifactManager:
    """Manages the creation, saving, and inspection of MLFlux run artifacts."""

    def __init__(self, base_output_dir: Union[str, Path] = "./mlflux_runs"):
        self.base_output_dir = Path(base_output_dir)

    def generate_run_id(self) -> str:
        """Create a deterministic-friendly, timestamped unique run ID."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:6]
        return f"run_{ts}_{uid}"

    def save_run_artifacts(
        self,
        run_id: str,
        pipeline: Pipeline,
        dataset_summary: Dict[str, Any],
        target_column: str,
        task_type: str,
        training_mode: str,
        primary_metric: str,
        best_model_name: str,
        best_cv_score: float,
        test_score: float,
        test_metrics: Dict[str, Any],
        leaderboard: List[Dict[str, Any]],
        feature_importance: List[Dict[str, Any]],
        elapsed_time_s: float,
        random_seed: int = 42,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        test_predictions_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Save all artifacts for a completed pipeline run."""
        run_dir = self.base_output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 0. Save data splits if provided
        if train_df is not None:
            train_df.to_csv(run_dir / "train.csv", index=False)
        if test_df is not None:
            test_df.to_csv(run_dir / "test.csv", index=False)
        if test_predictions_df is not None:
            test_predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

        # 1. Save complete pipeline (reusable end-to-end)
        save_joblib(pipeline, run_dir / "pipeline.joblib")

        # 2. Save fitted estimator alone
        if "estimator" in pipeline.named_steps:
            save_joblib(pipeline.named_steps["estimator"], run_dir / "model.joblib")

        # 3. Save metrics
        all_metrics = {
            "primary_metric": primary_metric,
            "best_cv_score": best_cv_score,
            "test_score": test_score,
            "test_metrics": test_metrics,
        }
        save_json(all_metrics, run_dir / "metrics.json")

        # 4. Save leaderboard
        # Strip fitted objects if any before serializing
        clean_leaderboard = []
        for row in leaderboard:
            r = dict(row)
            if "pipeline" in r:
                del r["pipeline"]
            clean_leaderboard.append(r)
        save_json(clean_leaderboard, run_dir / "leaderboard.json")

        # 5. Save feature importance
        save_json(feature_importance, run_dir / "feature_importance.json")

        # 6. Save metadata
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mlpipe_version": __version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "dataset": dataset_summary,
            "target_column": target_column,
            "task_type": task_type,
            "training_mode": training_mode,
            "primary_metric": primary_metric,
            "best_model": best_model_name,
            "best_cv_score": best_cv_score,
            "test_score": test_score,
            "random_seed": random_seed,
            "elapsed_time_s": elapsed_time_s,
            "splits": {
                "train_samples": len(train_df) if train_df is not None else None,
                "test_samples": len(test_df) if test_df is not None else None,
                "train_file": "train.csv" if train_df is not None else None,
                "test_file": "test.csv" if test_df is not None else None,
                "predictions_file": "test_predictions.csv" if test_predictions_df is not None else None,
            },
        }
        save_json(metadata, run_dir / "metadata.json")

        # 7. Generate text report
        report_text = self._build_text_report(metadata, clean_leaderboard, test_metrics, feature_importance)
        with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        return run_dir

    def _build_text_report(
        self,
        meta: Dict[str, Any],
        leaderboard: List[Dict[str, Any]],
        test_metrics: Dict[str, Any],
        feature_importance: List[Dict[str, Any]],
    ) -> str:
        lines = [
            f"==================================================",
            f"MLFlux Run Report - {meta['run_id']}",
            f"==================================================",
            f"Timestamp:       {meta['timestamp']}",
            f"MLFlux Version:  {meta['mlpipe_version']}",
            f"Dataset:         {meta['dataset'].get('filename', 'unknown')} ({meta['dataset'].get('rows')} rows, {meta['dataset'].get('columns')} cols)",
            f"Dataset SHA256:  {meta['dataset'].get('sha256', 'unknown')}",
            f"Target Column:   {meta['target_column']}",
            f"Task Type:       {meta['task_type'].capitalize()}",
            f"Training Mode:   {meta['training_mode']}",
            f"Seed:            {meta['random_seed']}",
            f"Elapsed Time:    {meta['elapsed_time_s']:.2f}s",
            f"",
            f"BEST MODEL",
            f"--------------------------------------------------",
            f"Model:           {meta['best_model']}",
            f"Primary Metric:  {meta['primary_metric']}",
            f"Best CV Score:   {meta['best_cv_score']}",
            f"Test Score:      {meta['test_score']}",
            f"",
            f"MODEL LEADERBOARD",
            f"--------------------------------------------------",
        ]
        for row in leaderboard:
            status = row.get("status", "unknown")
            cv = row.get("cv_score", "N/A")
            test = row.get("test_score", "N/A")
            time_s = row.get("training_time_s", "N/A")
            lines.append(f"{row['model']:<25} | CV: {str(cv):<8} | Test: {str(test):<8} | {time_s}s | {status}")

        lines.extend([
            f"",
            f"TEST EVALUATION METRICS",
            f"--------------------------------------------------",
        ])
        for k, v in test_metrics.items():
            if k != "confusion_matrix":
                lines.append(f"{k:<20}: {v}")

        splits = meta.get("splits", {})
        if splits and splits.get("train_samples") is not None:
            lines.extend([
                f"",
                f"DATA SPLITS & HOLD-OUT TEST SET",
                f"--------------------------------------------------",
                f"Training Samples:   {splits.get('train_samples')} ({splits.get('train_file')})",
                f"Testing Samples:    {splits.get('test_samples')} ({splits.get('test_file')})",
                f"Test Predictions:   {splits.get('predictions_file')}",
            ])

        if feature_importance:
            lines.extend([
                f"",
                f"TOP FEATURE IMPORTANCES",
                f"--------------------------------------------------",
            ])
            for item in feature_importance[:10]:
                lines.append(f"{item['feature']:<30}: {item['importance']}")

        return "\n".join(lines)


def load_pipeline_artifact(path: Union[str, Path]) -> Pipeline:
    """
    Load a fitted pipeline from either a run directory or direct joblib file path.
    """
    p = Path(path).resolve()
    if p.is_dir():
        candidate_file = p / "pipeline.joblib"
        if candidate_file.exists():
            return load_joblib(candidate_file)
        raise ArtifactError(
            f"No 'pipeline.joblib' found in directory '{p}'.",
            "Provide the path to the pipeline.joblib file or a valid MLFlux run directory."
        )

    if not p.exists():
        raise ArtifactError(f"Pipeline file '{p}' does not exist.")

    return load_joblib(p)


def inspect_run_directory(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Inspect and return metadata and artifacts summary from a run directory."""
    dir_path = Path(run_dir).resolve()
    if not dir_path.is_dir():
        raise ArtifactError(f"Path '{dir_path}' is not a directory.")

    meta_file = dir_path / "metadata.json"
    if not meta_file.exists():
        raise ArtifactError(
            f"Directory '{dir_path.name}' does not contain metadata.json.",
            "Verify that this is a completed MLFlux run directory."
        )

    metadata = load_json(meta_file)

    # Check available artifacts
    available_artifacts = [f.name for f in dir_path.iterdir() if f.is_file()]
    metadata["artifacts"] = available_artifacts

    # Load metrics if available
    metrics_file = dir_path / "metrics.json"
    if metrics_file.exists():
        metadata["detailed_metrics"] = load_json(metrics_file)

    return metadata
