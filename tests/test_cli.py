"""Tests for MLPipe CLI commands."""

from pathlib import Path
from typer.testing import CliRunner

from mlpipe.cli.main import app

runner = CliRunner()


def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "MLPipe: Production-ready tabular ML automation" in res.output


def test_cli_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "MLPipe v0.1.0" in res.output


def test_cli_profile_human(sample_csv_file: Path):
    res = runner.invoke(app, ["profile", str(sample_csv_file)])
    assert res.exit_code == 0
    assert "MLPipe Dataset Profile" in res.output
    assert "Dataset Overview" in res.output
    assert "Column Details" in res.output
    assert "numeric" in res.output



def test_cli_profile_json(sample_csv_file: Path):
    res = runner.invoke(app, ["profile", str(sample_csv_file), "--format", "json"])
    assert res.exit_code == 0
    assert '"num_rows": 120' in res.output


def test_cli_validate_success(sample_csv_file: Path):
    res = runner.invoke(app, ["validate", str(sample_csv_file), "--target", "target"])
    assert res.exit_code == 0
    assert "Status: PASSED" in res.output
    assert "Classification" in res.output


def test_cli_validate_missing_target(sample_csv_file: Path):
    res = runner.invoke(app, ["validate", str(sample_csv_file), "--target", "unknown_col"])
    assert res.exit_code != 0
    assert "Status: FAILED" in res.output


def test_cli_train_and_predict_and_inspect(sample_csv_file: Path, tmp_path: Path):
    runs_dir = tmp_path / "test_runs"

    # 1. Train fast mode
    train_res = runner.invoke(app, [
        "train",
        str(sample_csv_file),
        "--target", "target",
        "--mode", "fast",
        "--output", str(runs_dir),
    ])
    assert train_res.exit_code == 0
    assert "Best Model" in train_res.output
    assert "Model Evaluation" in train_res.output
    assert "Pipeline saved" in train_res.output

    # Find created run directory
    created_runs = list(runs_dir.glob("run_*"))
    assert len(created_runs) == 1
    run_dir = created_runs[0]
    pipeline_file = run_dir / "pipeline.joblib"
    assert pipeline_file.exists()
    assert (run_dir / "train.csv").exists()
    assert (run_dir / "test.csv").exists()
    assert (run_dir / "test_predictions.csv").exists()

    # 2. Inspect run directory
    inspect_res = runner.invoke(app, ["inspect", str(run_dir)])
    assert inspect_res.exit_code == 0
    assert "Run ID" in inspect_res.output
    assert "Best Model" in inspect_res.output
    assert "train.csv" in inspect_res.output

    # 3. Predict on new data
    pred_out = tmp_path / "preds.csv"
    pred_res = runner.invoke(app, [
        "predict",
        str(pipeline_file),
        str(sample_csv_file),
        "--output", str(pred_out),
    ])
    assert pred_res.exit_code == 0
    assert "predictions generated" in pred_res.output
    assert pred_out.exists()


def test_cli_split_command(sample_csv_file: Path, tmp_path: Path):
    split_dir = tmp_path / "splits"
    res = runner.invoke(app, [
        "split",
        str(sample_csv_file),
        "--target", "target",
        "--test-size", "0.25",
        "--output-dir", str(split_dir),
    ])
    assert res.exit_code == 0
    assert "Data Splits Summary" in res.output
    assert (split_dir / "train.csv").exists()
    assert (split_dir / "test.csv").exists()

