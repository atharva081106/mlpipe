"""Tests for standalone Python code & Jupyter Notebook generator."""

import ast
import json
from pathlib import Path
import subprocess
import sys

from mlpipe.codegen.generator import generate_standalone_code, generate_standalone_notebook
from mlpipe.core.pipeline import Pipeline


def test_codegen_produces_valid_syntax(sample_csv_file: Path):
    pipe = Pipeline(target="target", mode="fast")
    res = pipe.fit(sample_csv_file)

    estimator = pipe._fitted_pipeline.named_steps["estimator"]

    code = generate_standalone_code(
        dataset_path=str(sample_csv_file),
        target_column="target",
        task_type="classification",
        model_name=res.best_model_name,
        estimator_params=estimator.get_params(),
        numeric_columns=["numeric_feat", "skewed_feat"],
        categorical_columns=["cat_feat", "bool_feat"],
    )

    # 1. Verify AST syntax parse
    tree = ast.parse(code)
    assert tree is not None

    # 2. Verify expected keywords and VS Code interactive cell structure
    assert "# %%" in code
    assert "def main():" in code
    assert "ColumnTransformer" in code
    assert "StandardScaler" in code
    assert "OneHotEncoder" in code
    assert "joblib.dump" in code


def test_codegen_execution_end_to_end(sample_csv_file: Path, tmp_path: Path):
    pipe = Pipeline(target="target", mode="fast")
    res = pipe.fit(sample_csv_file)

    estimator = pipe._fitted_pipeline.named_steps["estimator"]

    code = generate_standalone_code(
        dataset_path=str(sample_csv_file),
        target_column="target",
        task_type="classification",
        model_name=res.best_model_name,
        estimator_params=estimator.get_params(),
        numeric_columns=["numeric_feat", "skewed_feat"],
        categorical_columns=["cat_feat", "bool_feat"],
    )

    script_path = tmp_path / "reproduce_script.py"
    script_path.write_text(code, encoding="utf-8")

    # Run the generated script with Python
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"Script failed with stderr: {proc.stderr}"
    assert "Training complete!" in proc.stdout
    assert "MODEL EVALUATION" in proc.stdout
    assert (tmp_path / "pipeline.joblib").exists()


def test_notebook_generation_produces_valid_ipynb(sample_csv_file: Path, tmp_path: Path):
    pipe = Pipeline(target="target", mode="fast")
    res = pipe.fit(sample_csv_file)

    estimator = pipe._fitted_pipeline.named_steps["estimator"]

    nb_content = generate_standalone_notebook(
        dataset_path=str(sample_csv_file),
        target_column="target",
        task_type="classification",
        model_name=res.best_model_name,
        estimator_params=estimator.get_params(),
        numeric_columns=["numeric_feat", "skewed_feat"],
        categorical_columns=["cat_feat", "bool_feat"],
    )

    # 1. Verify JSON validity
    nb_dict = json.loads(nb_content)
    assert "cells" in nb_dict
    assert nb_dict["nbformat"] == 4
    assert len(nb_dict["cells"]) > 5

    # 2. Check for markdown and code cells
    types = [c["cell_type"] for c in nb_dict["cells"]]
    assert "markdown" in types
    assert "code" in types

    # 3. Save as .ipynb file
    nb_path = tmp_path / "pipeline.ipynb"
    nb_path.write_text(nb_content, encoding="utf-8")
    assert nb_path.exists()
