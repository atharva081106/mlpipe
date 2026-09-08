"""Tests for MLFlux CLI commands."""

from pathlib import Path
from typer.testing import CliRunner

from mlflux.cli.main import app

runner = CliRunner()


def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "MLFlux: Production-ready tabular ML automation" in res.output


from mlflux.version import __version__


def test_cli_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert f"MLFlux v{__version__}" in res.output


def test_cli_profile_human(sample_csv_file: Path):
    res = runner.invoke(app, ["profile", str(sample_csv_file)])
    assert res.exit_code == 0
    assert "MLFlux Dataset Profile" in res.output
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


def test_cli_guided_run_interactive(sample_csv_file: Path, tmp_path: Path):
    script_out = tmp_path / "my_reproduce.py"
    # 1. target: "target"
    # 2. models: "1, 2"
    # 3. finetune: "n"
    # 4. export code: "y"
    # 5. script name: str(script_out)
    user_inputs = f"target\n1, 2\nn\ny\n{script_out}\n"

    res = runner.invoke(app, ["run", str(sample_csv_file)], input=user_inputs)
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "MLFlux Guided Studio" in res.output
    assert "Available Columns & Data Types" in res.output
    assert "Automated Exploratory Data Analysis" in res.output
    assert "Winning Model Selected" in res.output
    assert "Hold-Out Test Set Verification Preview" in res.output
    assert script_out.exists()


def test_cli_guided_run_finetune(sample_csv_file: Path, tmp_path: Path):
    script_out = tmp_path / "finetuned_reproduce.py"
    # 1. target: "target"
    # 2. models: "1" (single model)
    # 3. finetune: "y"
    # 4. keep parameter default (press Enter)
    user_inputs = "target\n1\ny\n\n"

    res = runner.invoke(app, ["run", str(sample_csv_file), "--export-code", str(script_out)], input=user_inputs)
    assert res.exit_code == 0, f"Command failed: {res.output}"
    assert "Fine-Tuning Options" in res.output
    assert script_out.exists()



