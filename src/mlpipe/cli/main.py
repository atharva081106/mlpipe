"""
Terminal Command Line Interface for MLPipe.

Built with Typer and Rich to deliver a polished, developer-focused terminal experience.
"""

import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
import typer

from mlpipe.artifacts.manager import inspect_run_directory
from mlpipe.codegen.generator import generate_standalone_code
from mlpipe.core.config import TaskType, TrainingMode
from mlpipe.core.exceptions import MLPipeError
from mlpipe.core.pipeline import Pipeline
from mlpipe.core.recommendations import (
    recommend_hyperparameter_overrides,
    recommend_models_for_task,
    recommend_split_strategy,
    recommend_target_column,
)
from mlpipe.data.eda import perform_eda
from mlpipe.data.ingestion import load_dataset
from mlpipe.data.profiling import profile_dataset
from mlpipe.data.splitting import split_data
from mlpipe.data.validation import detect_task, validate_dataset
from mlpipe.models.selection import get_candidates_for_task
from mlpipe.version import __version__

# Cross-platform encoding-safe glyphs
def _can_encode(char: str) -> bool:
    try:
        char.encode(sys.stdout.encoding or "utf-8")
        return True
    except Exception:
        return False

_USE_UNICODE = _can_encode("✓") and _can_encode("—")

CHECK = "✓" if _USE_UNICODE else "[OK]"
CROSS = "✗" if _USE_UNICODE else "[X]"
WARN = "⚠" if _USE_UNICODE else "[!]"
DASH = "—" if _USE_UNICODE else "-"
ARROW = "→" if _USE_UNICODE else "->"
BULLET = "•" if _USE_UNICODE else "*"
RULE = "─" if _USE_UNICODE else "-"


def _parse_param_override(val_str: str, original_val: Any = None) -> Any:
    val_clean = val_str.strip()
    if val_clean.lower() in ["none", "null"]:
        return None
    if val_clean.lower() == "true":
        return True
    if val_clean.lower() == "false":
        return False
    try:
        val_int = int(val_clean)
        if isinstance(original_val, (float, np.floating)):
            return float(val_int)
        return val_int
    except ValueError:
        pass
    try:
        return float(val_clean)
    except ValueError:
        pass
    return val_clean


app = typer.Typer(
    name="mlpipe",
    help="MLPipe: Production-ready tabular ML automation library and terminal CLI.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]MLPipe[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show MLPipe version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
):
    """Automate the machine learning lifecycle directly from the terminal."""
    pass


@app.command("version")
def version_cmd():
    """Display the installed MLPipe version."""
    console.print(f"[bold cyan]MLPipe[/bold cyan] v[green]{__version__}[/green]")


# ─── 1. Profile Command ───────────────────────────────────────────────────────

@app.command("profile")
def profile_cmd(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to the CSV dataset.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    format: str = typer.Option(
        "human",
        "--format",
        "-f",
        help="Output format: 'human' or 'json'.",
    ),
):
    """Profile a dataset and view structure, data types, missingness, and distributions."""
    try:
        ds = load_dataset(dataset_path)
        profile = profile_dataset(ds)

        if format.lower() == "json":
            print(profile.to_json())
            return

        # Human-readable output
        console.print(f"\n[bold cyan]MLPipe Dataset Profile[/bold cyan] {DASH} [bold]{ds.filename}[/bold]\n")

        # Overview Table
        ov_table = Table(title="Dataset Overview", show_header=False, border_style="dim")
        ov_table.add_row("Rows", f"{profile.num_rows:,}")
        ov_table.add_row("Columns", f"{profile.num_cols:,}")
        ov_table.add_row("Memory Usage", f"{profile.memory_mb} MB")
        ov_table.add_row("Duplicate Rows", f"{profile.duplicate_rows:,}")
        ov_table.add_row("Total Missing Values", f"{profile.total_missing_values:,}")
        console.print(ov_table)
        console.print()

        # Columns Table
        col_table = Table(title="Column Details", header_style="bold magenta")
        col_table.add_column("Column", style="cyan")
        col_table.add_column("Type", style="yellow")
        col_table.add_column("Missing", justify="right")
        col_table.add_column("Missing %", justify="right")
        col_table.add_column("Unique", justify="right")
        col_table.add_column("Stats / Top Values", style="dim")
        col_table.add_column("Flags", style="red")

        for c in profile.columns:
            stats_str = ""
            if c.detected_type == "numeric":
                stats_str = f"min: {c.min_val}, max: {c.max_val}, mean: {c.mean_val}"
            elif c.top_values:
                stats_str = ", ".join(f"{v['value']} ({v['count']})" for v in c.top_values[:3])
            elif c.min_date:
                stats_str = f"{c.min_date} {ARROW} {c.max_date}"

            flags_str = "; ".join(c.flags) if c.flags else DASH

            col_table.add_row(
                c.name,
                c.detected_type,
                f"{c.missing_count:,}",
                f"{c.missing_pct:.1f}%",
                f"{c.unique_count:,}",
                stats_str,
                flags_str,
            )

        console.print(col_table)

        if profile.warnings:
            console.print("\n[bold yellow]Dataset Warnings:[/bold yellow]")
            for w in profile.warnings:
                console.print(f"  [yellow]{WARN}[/yellow] {w}")
        console.print()

    except MLPipeError as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ─── 2. Validate Command ──────────────────────────────────────────────────────

@app.command("validate")
def validate_cmd(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to the CSV dataset.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Name of the target column to predict.",
    ),
    task: str = typer.Option(
        "auto",
        "--task",
        help="Task type override ('auto', 'classification', 'regression').",
    ),
    format: str = typer.Option(
        "human",
        "--format",
        "-f",
        help="Output format: 'human' or 'json'.",
    ),
):
    """Validate dataset suitability and verify pre-training feasibility."""
    try:
        ds = load_dataset(dataset_path)
        report = validate_dataset(ds, target_column=target, task_override=task)

        if format.lower() == "json":
            print(report.to_json())
            if not report.is_valid:
                raise typer.Exit(code=1)
            return

        console.print(f"\n[bold cyan]MLPipe Data Validation[/bold cyan] {DASH} [bold]{ds.filename}[/bold]\n")

        # Status
        status_color = "green" if report.is_valid else "red"
        status_text = "PASSED" if report.is_valid else "FAILED"
        console.print(f"Status: [bold {status_color}]{status_text}[/bold {status_color}]")
        console.print(f"Detected Task: [bold]{report.detected_task.capitalize()}[/bold]")
        console.print(f"Target Column: [bold]{target}[/bold]\n")

        if report.errors:
            console.print("[bold red]Fatal Errors:[/bold red]")
            for err in report.errors:
                console.print(f"  [bold red]{CROSS}[/bold red] {err}")
            console.print()

        if report.warnings:
            console.print("[bold yellow]Warnings (Non-blocking):[/bold yellow]")
            for warn in report.warnings:
                console.print(f"  [bold yellow]{WARN}[/bold yellow] {warn}")
            console.print()

        if report.is_valid:
            console.print(f"[bold green]{CHECK} Dataset is ready for automated training.[/bold green]\n")
        else:
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except MLPipeError as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ─── 3. Train Command ─────────────────────────────────────────────────────────

@app.command("train")
def train_cmd(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to the CSV dataset to train on.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Name of the target column to predict.",
    ),
    task: str = typer.Option(
        "auto",
        "--task",
        help="Task type: 'auto', 'classification', or 'regression'.",
    ),
    mode: str = typer.Option(
        "balanced",
        "--mode",
        "-m",
        help="Training mode budget: 'fast', 'balanced', or 'thorough'.",
    ),
    output: Path = typer.Option(
        Path("./mlpipe_runs"),
        "--output",
        "-o",
        help="Base directory to save run artifacts.",
    ),
    format: str = typer.Option(
        "human",
        "--format",
        "-f",
        help="Output format: 'human' or 'json'.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable detailed logging.",
    ),
    export_splits: Optional[Path] = typer.Option(
        None,
        "--export-splits",
        "-s",
        help="Optional destination directory to copy train.csv, test.csv, and test_predictions.csv.",
    ),
):
    """Train multiple candidate ML models, tune hyperparameters, evaluate, and save artifacts."""
    try:
        pipeline = Pipeline(
            target=target,
            task=task,
            mode=mode,
            output_dir=output,
            verbose=verbose,
        )

        if format.lower() != "json":
            console.print(f"\n[bold cyan]MLPipe[/bold cyan] v[green]{__version__}[/green]\n")

            # Load dataset for display header
            ds = load_dataset(dataset_path)

            console.print("[bold]Dataset[/bold]")
            console.print(RULE * 36)
            console.print(f"File       {ds.filename}")
            console.print(f"Rows       {ds.num_rows:,}")
            console.print(f"Columns    {ds.num_cols:,}")
            console.print(f"Target     {target}\n")

            def progress_hook(stage: str, msg: str):
                if stage == "ingestion":
                    console.print(f"[green]{CHECK}[/green] Dataset loaded")
                elif stage == "profiling":
                    console.print(f"[green]{CHECK}[/green] Dataset profiled")
                elif stage == "validation":
                    console.print(f"[green]{CHECK}[/green] Validation passed")
                elif stage == "task_detection":
                    console.print(f"[green]{CHECK}[/green] {msg}")
                elif stage == "splitting":
                    console.print(f"[green]{CHECK}[/green] Data split into Training & Testing sets")
                elif stage == "preprocessing":
                    console.print("\n[bold]Preprocessing[/bold]")
                    console.print(f"[green]{CHECK}[/green] Numeric features")
                    console.print(f"[green]{CHECK}[/green] Categorical features")
                    console.print(f"[green]{CHECK}[/green] Missing-value handling\n")
                    console.print("[bold]Training[/bold]")
                    console.print(RULE * 36)
                elif stage == "training":
                    model_part = msg.replace("Training & tuning candidate: ", "").replace("...", "")
                    console.print(f"[green]{CHECK}[/green] {model_part}")

            result = pipeline.fit(ds, on_progress=progress_hook)

            # Leaderboard Table
            console.print("\n[bold]Model Evaluation[/bold]")
            console.print(RULE * 36)

            lb_table = Table(header_style="bold cyan", border_style="dim")
            lb_table.add_column("Model", style="bold")
            lb_table.add_column(f"CV ({result.primary_metric})", justify="right", style="yellow")
            lb_table.add_column(f"Test ({result.primary_metric})", justify="right", style="green")
            lb_table.add_column("Time", justify="right", style="dim")
            lb_table.add_column("Status")

            for row in result.leaderboard:
                cv_val = f"{row['cv_score']:.4f}" if row['cv_score'] is not None else DASH
                test_val = f"{row['test_score']:.4f}" if row['test_score'] is not None else DASH
                time_val = f"{row['training_time_s']:.1f}s"
                status_val = f"[green]{CHECK}[/green]" if row['status'] == "success" else f"[red]{CROSS} ({row.get('error', 'fail')})[/red]"

                lb_table.add_row(row["model"], cv_val, test_val, time_val, status_val)

            console.print(lb_table)
            console.print(f"\n[green]{CHECK} Best model selected[/green]\n")

            # Best Model Panel
            panel_content = (
                f"[bold green]{result.best_model_name}[/bold green]\n"
                f"Primary metric: [bold]{result.primary_metric}[/bold]\n"
                f"CV Score:       [bold]{result.best_cv_score:.4f}[/bold]\n"
                f"Test Score:     [bold]{result.test_score:.4f}[/bold]"
            )
            console.print(Panel(panel_content, title="Best Model", border_style="green"))

            # Hold-Out Test Set Verification Preview (Results displayed in front of the user)
            if result.test_preview:
                console.print(f"\n[bold]Hold-Out Test Set Verification Preview[/bold] (First {len(result.test_preview)} rows)")
                console.print(RULE * 44)

                test_table = Table(header_style="bold cyan", border_style="dim")
                test_table.add_column("Row", style="dim", justify="right")
                test_table.add_column("Sample Features", style="cyan")
                test_table.add_column(f"Actual ({target})", justify="right", style="bold yellow")
                test_table.add_column(f"Predicted ({target})", justify="right", style="bold green")
                test_table.add_column("Evaluation", justify="center")

                for row_data in result.test_preview:
                    feat_str = ", ".join(f"{k}={v}" for k, v in row_data.get("features", {}).items())
                    act_val = row_data["actual"]
                    pred_val = row_data["predicted"]

                    if result.task_type == "classification":
                        match = row_data.get("match", False)
                        eval_str = f"[bold green]{CHECK} Match[/bold green]" if match else f"[bold red]{CROSS} Mismatch[/bold red]"
                        test_table.add_row(f"#{row_data['row_idx']}", feat_str, str(act_val), str(pred_val), eval_str)
                    else:
                        err = row_data.get("error", 0.0)
                        eval_str = f"Diff: {err:,.2f}"
                        test_table.add_row(
                            f"#{row_data['row_idx']}",
                            feat_str,
                            f"{act_val:,.2f}" if isinstance(act_val, (int, float)) else str(act_val),
                            f"{pred_val:,.2f}" if isinstance(pred_val, (int, float)) else str(pred_val),
                            eval_str,
                        )

                console.print(test_table)

            # Generated Datasets and Artifacts
            console.print("\n[bold]Generated Datasets & Artifacts[/bold]")
            console.print(RULE * 44)
            if result.train_path and result.train_path.exists():
                console.print(f"[green]{CHECK}[/green] Training dataset:   [bold cyan]{result.train_path.name}[/bold cyan] ({result.metadata.get('train_rows', 0):,} rows)")
            if result.test_path and result.test_path.exists():
                console.print(f"[green]{CHECK}[/green] Testing dataset:    [bold cyan]{result.test_path.name}[/bold cyan] ({result.metadata.get('test_rows', 0):,} rows)")
            if result.test_predictions_path and result.test_predictions_path.exists():
                console.print(f"[green]{CHECK}[/green] Test predictions:  [bold cyan]{result.test_predictions_path.name}[/bold cyan] (features + actual + predicted)")
            console.print(f"[green]{CHECK}[/green] Pipeline saved:     [bold cyan]pipeline.joblib[/bold cyan]")
            console.print(f"[green]{CHECK}[/green] Model saved:        [bold cyan]model.joblib[/bold cyan]")
            console.print(f"[green]{CHECK}[/green] Metrics saved:      [bold cyan]metrics.json[/bold cyan]")
            console.print(f"[green]{CHECK}[/green] Run report:         [bold cyan]report.txt[/bold cyan]\n")

            if export_splits:
                import shutil
                export_splits.mkdir(parents=True, exist_ok=True)
                if result.train_path and result.train_path.exists():
                    shutil.copy(result.train_path, export_splits / "train.csv")
                if result.test_path and result.test_path.exists():
                    shutil.copy(result.test_path, export_splits / "test.csv")
                if result.test_predictions_path and result.test_predictions_path.exists():
                    shutil.copy(result.test_predictions_path, export_splits / "test_predictions.csv")
                console.print(f"[green]{CHECK}[/green] Exported splits copied to: [bold]{export_splits}[/bold]\n")

            console.print(f"[bold]Output Directory:[/bold]\n{result.artifacts_dir}\n")

        else:
            # JSON format
            result = pipeline.fit(dataset_path)
            print(json.dumps(result.to_dict(), indent=2))

    except MLPipeError as e:
        err_console.print(f"\n[bold red]MLPipe Error:[/bold red]\n{e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"\n[bold red]Unexpected Error:[/bold red] {e}")
        if verbose:
            err_console.print_exception()
        raise typer.Exit(code=1)


# ─── 4. Predict Command ───────────────────────────────────────────────────────

@app.command("predict")
def predict_cmd(
    model_path: Path = typer.Argument(
        ...,
        help="Path to pipeline.joblib or run directory.",
        exists=True,
    ),
    data_path: Path = typer.Argument(
        ...,
        help="Path to the new CSV data for predictions.",
        exists=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to save predictions as a CSV.",
    ),
):
    """Generate predictions on new data using a trained MLPipe pipeline."""
    try:
        pipeline = Pipeline.load(model_path)
        predictions = pipeline.predict(data_path)

        console.print("\n[bold cyan]Predictions[/bold cyan]")
        console.print(RULE * 24)

        # Show up to first 10 predictions
        sample_size = min(10, len(predictions))
        for i in range(sample_size):
            console.print(f"Row {i+1:<4}  [bold]{predictions[i]}[/bold]")

        if len(predictions) > sample_size:
            console.print(f"[dim]... and {len(predictions) - sample_size:,} more rows[/dim]")

        console.print(f"\n[green]{CHECK}[/green] [bold]{len(predictions):,}[/bold] predictions generated.\n")

        if output:
            out_df = pd.DataFrame({"prediction": predictions})
            out_df.to_csv(output, index=False)
            console.print(f"[green]{CHECK}[/green] Saved predictions to [bold]{output}[/bold]\n")

    except MLPipeError as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ─── 5. Inspect Command ───────────────────────────────────────────────────────

@app.command("inspect")
def inspect_cmd(
    run_dir: Path = typer.Argument(
        ...,
        help="Path to the MLPipe run directory (e.g. ./mlpipe_runs/<run_id>).",
        exists=True,
        file_okay=False,
    ),
    format: str = typer.Option(
        "human",
        "--format",
        "-f",
        help="Output format: 'human' or 'json'.",
    ),
):
    """Inspect previous run metadata, leaderboard, and artifacts."""
    try:
        meta = inspect_run_directory(run_dir)

        if format.lower() == "json":
            print(json.dumps(meta, indent=2))
            return

        console.print(f"\n[bold cyan]MLPipe Run Inspection[/bold cyan] {DASH} [bold]{meta.get('run_id')}[/bold]\n")

        table = Table(title="Run Summary", show_header=False, border_style="dim")
        table.add_row("Run ID", str(meta.get("run_id")))
        table.add_row("Timestamp", str(meta.get("timestamp")))
        table.add_row("Dataset", f"{meta.get('dataset', {}).get('filename')} ({meta.get('dataset', {}).get('rows')} rows)")
        table.add_row("Target", str(meta.get("target_column")))
        table.add_row("Task", str(meta.get("task_type")).capitalize())
        table.add_row("Best Model", f"[bold green]{meta.get('best_model')}[/bold green]")
        table.add_row("Primary Metric", str(meta.get("primary_metric")))
        table.add_row("CV Score", str(meta.get("best_cv_score")))
        table.add_row("Test Score", str(meta.get("test_score")))
        table.add_row("Training Mode", str(meta.get("training_mode")))
        table.add_row("Seed", str(meta.get("random_seed")))
        table.add_row("Elapsed Time", f"{meta.get('elapsed_time_s', 0):.2f}s")

        if "splits" in meta and meta["splits"].get("train_samples") is not None:
            sp = meta["splits"]
            table.add_row("Training Split", f"{sp['train_samples']:,} rows ({sp.get('train_file')})")
            table.add_row("Testing Split", f"{sp['test_samples']:,} rows ({sp.get('test_file')})")
            table.add_row("Predictions File", str(sp.get("predictions_file")))

        console.print(table)
        console.print()

        # Available Artifacts
        artifacts = meta.get("artifacts", [])
        console.print("[bold]Generated Artifacts:[/bold]")
        for art in artifacts:
            console.print(f"  [green]{BULLET}[/green] {art}")
        console.print()

    except MLPipeError as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ─── 6. Split Command ─────────────────────────────────────────────────────────

@app.command("split")
def split_cmd(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to the CSV dataset to split.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target column to stratify (for classification) and preserve in splits.",
    ),
    test_size: float = typer.Option(
        0.20,
        "--test-size",
        "-s",
        help="Fraction of data for the holdout test set (default: 0.20).",
    ),
    output_dir: Path = typer.Option(
        Path("./splits"),
        "--output-dir",
        "-o",
        help="Directory to save train.csv and test.csv.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducible splitting.",
    ),
):
    """Explicitly split a dataset into train.csv and test.csv without data leakage."""
    try:
        ds = load_dataset(dataset_path)
        if target not in ds.data.columns:
            raise MLPipeError(f"Target column '{target}' not found in dataset columns: {list(ds.data.columns)}")

        task_type = detect_task(ds.data[target])
        split_res = split_data(
            df=ds.data,
            target_column=target,
            task_type=task_type,
            test_size=test_size,
            random_seed=seed,
        )

        train_file, test_file = split_res.export(output_dir)

        console.print(f"\n[bold cyan]MLPipe Dataset Split[/bold cyan] {DASH} [bold]{ds.filename}[/bold]\n")

        table = Table(title="Data Splits Summary", header_style="bold cyan", border_style="dim")
        table.add_column("Set", style="bold")
        table.add_column("Rows", justify="right")
        table.add_column("Percentage", justify="right")
        table.add_column("Stratified", justify="center")
        table.add_column("Saved Path", style="dim")

        table.add_row(
            "Training Set",
            f"{split_res.train_size:,}",
            f"{100*(1-test_size):.1f}%",
            f"[green]{CHECK}[/green]" if split_res.is_stratified else DASH,
            str(train_file.resolve()),
        )
        table.add_row(
            "Testing Set",
            f"{split_res.test_size:,}",
            f"{100*test_size:.1f}%",
            f"[green]{CHECK}[/green]" if split_res.is_stratified else DASH,
            str(test_file.resolve()),
        )

        console.print(table)

        # Show preview of test set rows right in front of user
        console.print(f"\n[bold]Testing Set Preview[/bold] (First 5 hold-out rows)")
        console.print(RULE * 36)
        preview_cols = [c for c in split_res.test_df.columns if c != target][:3] + [target]
        prev_table = Table(header_style="bold magenta", border_style="dim")
        for c in preview_cols:
            prev_table.add_column(c, style="bold yellow" if c == target else "white")
        for _, row in split_res.test_df.head(5).iterrows():
            prev_table.add_row(*[str(row[c]) for c in preview_cols])
        console.print(prev_table)

        console.print(f"\n[green]{CHECK}[/green] Datasets successfully exported to [bold]{output_dir}[/bold]\n")

    except MLPipeError as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ─── 7. Interactive Guided Studio Command ─────────────────────────────────────

@app.command("run")
@app.command("interactive")
def run_cmd(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to the CSV dataset to analyze and train on.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("./mlpipe_runs"),
        "--output",
        "-o",
        help="Base directory to save run artifacts.",
    ),
    export_code: Optional[Path] = typer.Option(
        None,
        "--export-code",
        "-c",
        help="Optional destination path to export standalone reproducible Python script.",
    ),
):
    """
    Interactive guided studio: analyze CSV, choose target factor, view EDA,
    select ML models, evaluate test results, fine-tune parameters, and export reproducible code.
    """
    try:
        console.print(f"\n[bold cyan]MLPipe Guided Studio[/bold cyan] {DASH} [bold]{dataset_path.name}[/bold]\n")

        # ── Step 1: Ingest & Inspect Columns ──────────────────────────────────────
        ds = load_dataset(dataset_path)

        ov_table = Table(title="Dataset Overview", show_header=False, border_style="dim")
        ov_table.add_row("Filename", ds.filename)
        ov_table.add_row("Total Rows", f"{ds.num_rows:,}")
        ov_table.add_row("Total Columns", f"{ds.num_cols:,}")
        ov_table.add_row("Memory Size", f"{ds.memory_mb} MB")
        console.print(ov_table)
        console.print()

        col_table = Table(title="Available Columns & Data Types", header_style="bold cyan", border_style="dim")
        col_table.add_column("#", justify="right", style="bold yellow")
        col_table.add_column("Column Name", style="bold")
        col_table.add_column("Type", style="cyan")
        col_table.add_column("Missing %", justify="right")
        col_table.add_column("Unique", justify="right")
        col_table.add_column("Sample Values", style="dim")

        cols = list(ds.df.columns)
        for idx, col in enumerate(cols, start=1):
            dtype = str(ds.df[col].dtype)
            missing_pct = round((ds.df[col].isnull().sum() / len(ds.df)) * 100, 1)
            unique_cnt = ds.df[col].nunique()
            samples = ", ".join(str(v) for v in ds.df[col].dropna().unique()[:3])
            col_table.add_row(f"[{idx}]", col, dtype, f"{missing_pct}%", f"{unique_cnt:,}", samples[:35])

        console.print(col_table)

        # ── Step 2: Choose Target Factor to Predict ───────────────────────────────
        rec_target, rec_reason = recommend_target_column(ds.df)
        rec_target_idx = cols.index(rec_target) + 1

        console.print(Panel(
            f"💡 [bold green]Best Recommendation:[/bold green] Column [[bold yellow]{rec_target_idx}[/bold yellow]] [bold cyan]'{rec_target}'[/bold cyan]\n"
            f"[dim]Rationale: {rec_reason}[/dim]",
            title="Recommended Target Factor",
            border_style="green",
        ))

        target_choice = Prompt.ask(
            "\n[bold green]Which factor / column do you want to predict?[/bold green] (enter number or name, press Enter for recommended)",
            default=str(rec_target_idx),
        )

        target = None
        if target_choice.isdigit() and 1 <= int(target_choice) <= len(cols):
            target = cols[int(target_choice) - 1]
        elif target_choice in cols:
            target = target_choice
        else:
            matched = [c for c in cols if c.lower() == target_choice.lower()]
            if matched:
                target = matched[0]
            else:
                err_console.print(f"[bold red]Column '{target_choice}' not found in dataset columns.[/bold red]")
                raise typer.Exit(code=1)

        task_type = detect_task(ds.df[target])
        console.print(f"\n[green]{CHECK}[/green] Target Selected: [bold yellow]{target}[/bold yellow]")
        console.print(f"[green]{CHECK}[/green] Inferred Problem Type: [bold cyan]{task_type.capitalize()}[/bold cyan]")

        # ── Step 3: Automated EDA & Splitting ─────────────────────────────────────
        console.print("\n[bold]Automated Exploratory Data Analysis (EDA)[/bold]")
        console.print(RULE * 44)
        eda_res = perform_eda(ds.df, target, task_type)

        if task_type == "classification":
            tdist_table = Table(title=f"Target Class Distribution ('{target}')", header_style="bold cyan", border_style="dim")
            tdist_table.add_column("Class Label", style="bold")
            tdist_table.add_column("Count", justify="right")
            tdist_table.add_column("Percentage", justify="right")
            for row in eda_res.target_distribution.get("classes", []):
                tdist_table.add_row(str(row["class"]), f"{row['count']:,}", f"{row['percentage']}%")
            console.print(tdist_table)
        else:
            tdist_table = Table(title=f"Target Numerical Distribution ('{target}')", show_header=False, border_style="dim")
            for k, v in eda_res.target_distribution.items():
                tdist_table.add_row(k.capitalize(), str(v))
            console.print(tdist_table)

        if eda_res.correlations:
            corr_table = Table(title=f"Top Correlated Features with '{target}'", header_style="bold magenta", border_style="dim")
            corr_table.add_column("Feature", style="bold")
            corr_table.add_column("Abs. Correlation", justify="right", style="yellow")
            for c in eda_res.correlations:
                corr_table.add_row(c["feature"], str(c["correlation"]))
            console.print(corr_table)

        split_rec = recommend_split_strategy(len(ds.df), task_type)
        console.print(f"\n💡 [bold green]Recommended Split Strategy:[/bold green] {split_rec['explanation']}")
        console.print(f"   [dim]{split_rec['stratification_note']}[/dim]")

        split_data_res = split_data(ds.df, target, task_type, test_size=split_rec["test_size"])
        console.print(f"[green]{CHECK}[/green] Data Split Applied: [bold]{split_data_res.train_size:,}[/bold] train rows / [bold]{split_data_res.test_size:,}[/bold] test rows ({int(100*(1-split_rec['test_size']))}% / {int(100*split_rec['test_size'])}%)")

        # ── Step 4: Model Choice (Single or Multiple) ─────────────────────────────
        candidates = get_candidates_for_task(task_type, mode="balanced")
        cand_names = [c.name for c in candidates]
        rec_models = recommend_models_for_task(task_type, len(ds.df), cand_names)

        console.print(f"\n[bold]Select Machine Learning Models to Train[/bold]")
        console.print(RULE * 44)

        for idx, cand in enumerate(candidates, start=1):
            note = rec_models["candidate_notes"].get(cand.name, "")
            console.print(f"  [bold yellow][{idx}][/bold yellow] [bold]{cand.name}[/bold]")
            if note:
                console.print(f"      [dim]{note}[/dim]")

        console.print(f"  [bold yellow][A][/bold yellow] [bold]All Models[/bold] [green]⭐ (Recommended Benchmark)[/green]")
        console.print(f"      [dim]Trains, tunes, and compares all candidates with 5-fold CV to empirically crown the best model.[/dim]")

        console.print(Panel(
            f"💡 [bold green]Best Recommendation:[/bold green] [bold cyan]Option [A] (All Models)[/bold cyan]\n"
            f"[dim]{rec_models['all_recommendation']}[/dim]",
            title="Model Selection Recommendation",
            border_style="cyan",
        ))

        model_choice = Prompt.ask(
            "\n[bold green]Which models would you like to train?[/bold green] (enter numbers e.g. 1, 2 or press Enter for recommended 'A')",
            default=rec_models["recommended_choice"],
        )

        selected_model_names = None
        if model_choice.strip().lower() not in ["a", "all", "*"]:
            selected_indices = [
                int(p.strip()) for p in model_choice.replace(";", ",").split(",")
                if p.strip().isdigit() and 1 <= int(p.strip()) <= len(candidates)
            ]
            if selected_indices:
                selected_model_names = [candidates[i - 1].name for i in selected_indices]
                console.print(f"[green]{CHECK}[/green] Training selected models: [cyan]{', '.join(selected_model_names)}[/cyan]")
            else:
                console.print("[yellow]Invalid choice, defaulting to all candidate models.[/yellow]")

        if not selected_model_names:
            console.print(f"[green]{CHECK}[/green] Training all {len(candidates)} candidate models")

        # ── Step 5: Train, Test, and Present Results ──────────────────────────────
        console.print("\n[bold]Training & Evaluating Models[/bold]")
        console.print(RULE * 44)

        pipe = Pipeline(
            target=target,
            task=task_type,
            mode="balanced",
            selected_models=selected_model_names,
            output_dir=output,
        )

        def step_hook(stage: str, msg: str):
            if stage == "training":
                m = msg.replace("Training & tuning candidate: ", "").replace("...", "")
                console.print(f"  [green]{CHECK}[/green] Trained & tuned {m}")

        result = pipe.fit(ds, on_progress=step_hook)

        # Present leaderboard
        lb_table = Table(title="Model Evaluation Leaderboard", header_style="bold cyan", border_style="dim")
        lb_table.add_column("Model", style="bold")
        lb_table.add_column(f"CV ({result.primary_metric})", justify="right", style="yellow")
        lb_table.add_column(f"Test ({result.primary_metric})", justify="right", style="green")
        lb_table.add_column("Time", justify="right", style="dim")
        lb_table.add_column("Status")

        for row in result.leaderboard:
            cv_str = f"{row['cv_score']:.4f}" if row['cv_score'] is not None else DASH
            test_str = f"{row['test_score']:.4f}" if row['test_score'] is not None else DASH
            t_str = f"{row['training_time_s']:.1f}s"
            stat_str = f"[green]{CHECK}[/green]" if row['status'] == "success" else f"[red]{CROSS}[/red]"
            lb_table.add_row(row["model"], cv_str, test_str, t_str, stat_str)

        console.print(lb_table)

        # Winning Model Panel
        console.print(Panel(
            f"[bold green]{result.best_model_name}[/bold green]\n"
            f"Primary Metric: [bold]{result.primary_metric}[/bold]\n"
            f"Cross-Validation Score: [bold]{result.best_cv_score:.4f}[/bold]\n"
            f"Hold-out Test Score:    [bold]{result.test_score:.4f}[/bold]",
            title="Winning Model Selected",
            border_style="green"
        ))

        # Hold-Out Test Set Verification Preview
        if result.test_preview:
            console.print(f"\n[bold]Hold-Out Test Set Verification Preview[/bold] (Actual Ground Truth vs Model Prediction)")
            console.print(RULE * 44)
            test_table = Table(header_style="bold cyan", border_style="dim")
            test_table.add_column("Row", style="dim", justify="right")
            test_table.add_column("Sample Features", style="cyan")
            test_table.add_column(f"Actual ({target})", justify="right", style="bold yellow")
            test_table.add_column(f"Predicted ({target})", justify="right", style="bold green")
            test_table.add_column("Evaluation", justify="center")

            for row_data in result.test_preview:
                feat_str = ", ".join(f"{k}={v}" for k, v in row_data.get("features", {}).items())
                act_val = row_data["actual"]
                pred_val = row_data["predicted"]

                if result.task_type == "classification":
                    match = row_data.get("match", False)
                    eval_str = f"[bold green]{CHECK} Match[/bold green]" if match else f"[bold red]{CROSS} Mismatch[/bold red]"
                    test_table.add_row(f"#{row_data['row_idx']}", feat_str, str(act_val), str(pred_val), eval_str)
                else:
                    err = row_data.get("error", 0.0)
                    eval_str = f"Diff: {err:,.2f}"
                    test_table.add_row(
                        f"#{row_data['row_idx']}",
                        feat_str,
                        f"{act_val:,.2f}" if isinstance(act_val, (int, float)) else str(act_val),
                        f"{pred_val:,.2f}" if isinstance(pred_val, (int, float)) else str(pred_val),
                        eval_str,
                    )

            console.print(test_table)

        # ── Step 6: Interactive Parameter Fine-Tuning ─────────────────────────────
        console.print("\n[bold]Fine-Tuning Options[/bold]")
        console.print(RULE * 44)
        console.print(
            "💡 [bold green]Recommendation:[/bold green] [dim]Automated 5-fold CV hyperparameter search has already discovered "
            "an optimal configuration. Fine-tuning is optional for testing custom parameter ranges or constraints.[/dim]\n"
        )
        do_finetune = Confirm.ask(
            f"[bold green]Would you like to fine-tune the winning model ({result.best_model_name}) with custom parameters?[/bold green]",
            default=False,
        )

        if do_finetune and pipe._fitted_pipeline:
            best_estimator = pipe._fitted_pipeline.named_steps.get("estimator")
            if best_estimator:
                current_params = best_estimator.get_params()
                hp_recs = recommend_hyperparameter_overrides(result.best_model_name, current_params)
                tuneable_keys = [k for k in ["n_estimators", "max_depth", "learning_rate", "C", "min_samples_split", "n_neighbors", "alpha"] if k in current_params]

                console.print(f"\nCurrent parameters and [bold green]Recommended Overrides[/bold green] for [bold]{result.best_model_name}[/bold]:")
                for k in tuneable_keys:
                    if k in hp_recs:
                        rec_info = hp_recs[k]
                        console.print(f"  {BULLET} [bold]{k}[/bold]: Current = [yellow]{rec_info['current']}[/yellow] | 💡 [bold green]Recommended = {rec_info['recommended']}[/bold green] [dim]({rec_info['rationale']})[/dim]")
                    else:
                        console.print(f"  {BULLET} [bold]{k}[/bold]: Current = [yellow]{current_params[k]}[/yellow]")

                param_overrides = {}
                console.print("\nEnter new parameter values (or press Enter to accept recommended / current):")
                for k in tuneable_keys:
                    rec_val = hp_recs[k]["recommended"] if k in hp_recs else current_params[k]
                    val_input = Prompt.ask(f"  {k}", default=str(rec_val))
                    parsed_val = _parse_param_override(val_input, current_params[k])
                    if parsed_val != current_params[k]:
                        param_overrides[k] = parsed_val

                if param_overrides:
                    console.print(f"\n[cyan]Retuning with parameters: {param_overrides}...[/cyan]")
                    tune_res = pipe.retune_best_model(param_overrides)

                    cmp_table = Table(title="Fine-Tuning Performance Comparison", header_style="bold cyan", border_style="dim")
                    cmp_table.add_column("Stage", style="bold")
                    cmp_table.add_column(f"Test Score ({tune_res['primary_metric']})", justify="right")
                    cmp_table.add_column("Delta", justify="right")

                    diff = tune_res["new_test_score"] - tune_res["old_test_score"]
                    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                    diff_styled = f"[green]{diff_str}[/green]" if diff >= 0 else f"[red]{diff_str}[/red]"

                    cmp_table.add_row("Before Fine-Tuning", f"{tune_res['old_test_score']:.4f}", DASH)
                    cmp_table.add_row("After Fine-Tuning", f"{tune_res['new_test_score']:.4f}", diff_styled)
                    console.print(cmp_table)
                    console.print(f"[green]{CHECK}[/green] Model updated with fine-tuned parameters.")
                else:
                    console.print("[dim]No parameters changed.[/dim]")

        # ── Step 7: Export Standalone Python Script ───────────────────────────────
        console.print("\n[bold]Code Generation & Export[/bold]")
        console.print(RULE * 44)
        console.print(
            "💡 [bold green]Recommendation:[/bold green] [bold cyan]Yes (Recommended)[/bold cyan] — Generates a standalone, "
            "reproducible Python script (.py) using standard scikit-learn & pandas with zero external dependencies on mlpipe.\n"
        )
        want_code = export_code is not None or Confirm.ask(
            "[bold green]Would you like to get the complete, standalone Python code performed on this dataset?[/bold green]",
            default=True,
        )

        if want_code and pipe._fitted_pipeline:
            best_estimator = pipe._fitted_pipeline.named_steps.get("estimator")
            if best_estimator:
                num_cols = list(ds.df.select_dtypes(include=[np.number]).columns)
                cat_cols = [c for c in ds.df.columns if c not in num_cols and c != target]

                code = generate_standalone_code(
                    dataset_path=str(dataset_path.resolve()),
                    target_column=target,
                    task_type=task_type,
                    model_name=result.best_model_name,
                    estimator_params=best_estimator.get_params(),
                    numeric_columns=num_cols,
                    categorical_columns=cat_cols,
                    test_size=split_rec["test_size"],
                    random_seed=42,
                )

                if export_code:
                    script_path = export_code
                else:
                    console.print("💡 [bold green]Recommended script name:[/bold green] [bold cyan]reproduce_pipeline.py[/bold cyan]")
                    script_path_str = Prompt.ask("Save script as", default="reproduce_pipeline.py")
                    script_path = Path(script_path_str)

                script_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.write_text(code, encoding="utf-8")

                # Show code snippet
                console.print(f"\n[green]{CHECK}[/green] Standalone Python script written to [bold cyan]{script_path}[/bold cyan]\n")
                console.print("[bold]Script Preview (First 25 lines):[/bold]")
                preview_lines = "\n".join(code.splitlines()[:25])
                console.print(Syntax(preview_lines, "python", theme="monokai", line_numbers=True))
                console.print(f"\n[dim]Run this script independently on any machine with:[/dim]")
                console.print(f"[bold]python {script_path.name}[/bold]\n")

        console.print(f"[bold green]{CHECK} MLPipe Guided Session Complete![/bold green]\n")

    except MLPipeError as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(code=1)


