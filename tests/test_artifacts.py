"""Tests for artifact serialization and management."""

from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from mlflux.artifacts.manager import (
    ArtifactManager,
    inspect_run_directory,
    load_pipeline_artifact,
)


def test_artifact_manager_bundle_lifecycle(tmp_path: Path):
    manager = ArtifactManager(base_output_dir=tmp_path)
    run_id = manager.generate_run_id()

    dummy_pipe = Pipeline([("estimator", DummyClassifier())])
    dummy_pipe.fit([[1], [2]], [0, 1])

    saved_dir = manager.save_run_artifacts(
        run_id=run_id,
        pipeline=dummy_pipe,
        dataset_summary={"filename": "test.csv", "rows": 100, "columns": 5},
        target_column="target",
        task_type="classification",
        training_mode="balanced",
        primary_metric="f1_weighted",
        best_model_name="DummyClassifier",
        best_cv_score=0.75,
        test_score=0.72,
        test_metrics={"accuracy": 0.72},
        leaderboard=[{"model": "DummyClassifier", "cv_score": 0.75}],
        feature_importance=[],
        elapsed_time_s=1.5,
    )

    # Verify all expected artifact files exist
    assert (saved_dir / "pipeline.joblib").exists()
    assert (saved_dir / "model.joblib").exists()
    assert (saved_dir / "metrics.json").exists()
    assert (saved_dir / "leaderboard.json").exists()
    assert (saved_dir / "metadata.json").exists()
    assert (saved_dir / "feature_importance.json").exists()
    assert (saved_dir / "report.txt").exists()

    # Test loading pipeline from run directory
    loaded_pipe = load_pipeline_artifact(saved_dir)
    assert loaded_pipe is not None
    preds = loaded_pipe.predict([[1], [2]])
    assert len(preds) == 2

    # Test inspect
    metadata = inspect_run_directory(saved_dir)
    assert metadata["run_id"] == run_id
    assert metadata["best_model"] == "DummyClassifier"
    assert "pipeline.joblib" in metadata["artifacts"]
