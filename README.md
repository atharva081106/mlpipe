# MLPipe

> **Production-ready tabular ML automation library and terminal CLI.**  
> From raw CSV to evaluated, reproducible, deployable model pipelines in one command.

---

## 🚀 Overview

**MLPipe** automates the repetitive engineering lifecycle for tabular machine learning models. Simply point MLPipe at your CSV dataset and designate a target column:

```
Raw CSV Dataset
      ↓
Data Ingestion (Format validation & SHA-256 integrity hash)
      ↓
Data Profiling (Column types, distributions, missingness & flags)
      ↓
Pre-training Validation (Fatal structural errors & non-blocking warnings)
      ↓
Task Detection (Automatic classification vs. regression inference)
      ↓
Leakage-Free Splitting (Stratified or random train/test split)
      ↓
Automated Preprocessing (ColumnTransformer: imputation, scaling & encoding)
      ↓
Multi-Model Training & Tuning (RandomizedSearchCV strictly on training folds)
      ↓
CV-Based Leaderboard Ranking (Winning model selected by CV, not test set)
      ↓
Held-Out Test Evaluation (Unbiased final metrics & confusion matrix)
      ↓
Model Explainability (Recovered transformed feature importances & coefficients)
      ↓
Artifact Generation (Reusable pipeline, model, metadata & reports)
      ↓
Instant Predictions (CLI & Python inference)
```

No hardcoded results. No fake training. Real scikit-learn models and metrics computed on your CPU machine.

---

## 📦 Installation

Install locally in editable mode:

```bash
git clone https://github.com/your-org/mlpipe.git
cd mlpipe
pip install -e .
```

Verify the installation:

```bash
mlpipe --version
```

---

## ⚡ Quick Start

### 1. Terminal CLI

Train an end-to-end classification pipeline:

```bash
mlpipe train demo_data/customer_churn.csv --target churn
```

Train a regression pipeline:

```bash
mlpipe train demo_data/house_prices.csv --target price
```

Generate predictions on new data:

```bash
mlpipe predict ./mlpipe_runs/<run_id>/pipeline.joblib demo_data/customer_churn.csv --output predictions.csv
```

### 2. Python API

```python
from mlpipe import Pipeline

# 1. Initialize pipeline
pipeline = Pipeline(
    target="churn",
    task="auto",       # auto-detects classification vs regression
    mode="balanced",   # "fast", "balanced", or "thorough"
)

# 2. Fit pipeline on CSV or DataFrame
result = pipeline.fit("demo_data/customer_churn.csv")

# 3. Inspect results
print("Best Model:", result.best_model)
print("Primary Metric:", result.primary_metric)
print("CV Score:", result.best_cv_score)
print("Test Score:", result.test_score)
print("Test Metrics:", result.metrics)

# 4. Save and reload pipeline
pipeline.save("./trained_models/churn_model")

loaded_pipeline = Pipeline.load("./trained_models/churn_model")
predictions = loaded_pipeline.predict("demo_data/customer_churn.csv")
print("Predictions:", predictions[:5])
```

---

## 🛠️ CLI Command Reference

### `mlpipe train`
Train multiple candidate models, tune hyperparameters, rank on leaderboard, and save artifacts.

```bash
mlpipe train <data.csv> --target <target_col> [OPTIONS]
```

**Options:**
- `--target, -t`: Target column name to predict (required).
- `--task`: Task type override: `auto` (default), `classification`, or `regression`.
- `--mode, -m`: Training mode budget: `fast`, `balanced` (default), or `thorough`.
- `--output, -o`: Base directory to store run artifacts (default: `./mlpipe_runs`).
- `--format, -f`: Output format: `human` (default) or `json`.
- `--verbose`: Enable detailed debug logging.

### `mlpipe profile`
Inspect dataset summary, column types, statistics, missingness, and structural issues.

```bash
mlpipe profile demo_data/customer_churn.csv
mlpipe profile demo_data/customer_churn.csv --format json
```

### `mlpipe validate`
Run pre-training validation checks on dataset and target.

```bash
mlpipe validate demo_data/customer_churn.csv --target churn
```

### `mlpipe predict`
Generate predictions on a new CSV dataset using a saved pipeline.

```bash
mlpipe predict ./mlpipe_runs/<run_id>/pipeline.joblib new_data.csv --output preds.csv
```

### `mlpipe inspect`
Inspect metadata, configuration, metrics, and generated artifacts from a previous run directory.

```bash
mlpipe inspect ./mlpipe_runs/<run_id>
mlpipe inspect ./mlpipe_runs/<run_id> --format json
```

### `mlpipe version`
Display the current version of MLPipe.

```bash
mlpipe version
```

---

## 🧠 Supported Models

### Classification
- **Logistic Regression** (L2 penalty, liblinear/lbfgs/saga solvers)
- **Random Forest Classifier** (trees, depth, sample split/leaf tuning)
- **HistGradientBoosting Classifier** (iterations, learning rate, leaf bounds)
- **Decision Tree Classifier** (depth, split thresholds, criteria)
- **K-Nearest Neighbors** (neighbors, weights, distance metrics)

### Regression
- **Ridge Regression** (regularization alpha, solver selection)
- **Random Forest Regressor** (trees, depth, sample bounds, features)
- **HistGradientBoosting Regressor** (iterations, rate, regularization)
- **Decision Tree Regressor** (depth, split criteria, sample leaves)

---

## 🔒 Data Leakage Prevention Guarantee

MLPipe adheres to strict data integrity standards:
1. **No Full-Data Transformations:** Preprocessing pipelines are never fitted on the entire dataset.
2. **Train/Test Split First:** The raw dataset is partitioned (default 80% train, 20% test) before column classification and transformer fitting.
3. **Cross-Validation Inside Pipelines:** During hyperparameter search, sklearn `Pipeline` objects fit transformers solely on the internal training fold of each split.
4. **CV-Based Model Selection:** The winning model is selected strictly based on CV score on the training set. The held-out test set is evaluated exactly once for unbiased reporting.

---

## 📁 Artifact Structure

Each completed training run generates a self-contained bundle under `./mlpipe_runs/<run_id>/`:

```text
mlpipe_runs/<run_id>/
├── pipeline.joblib          # Complete fitted pipeline (preprocessor + estimator)
├── model.joblib             # Fitted estimator alone
├── metrics.json             # CV and test evaluation metrics
├── leaderboard.json         # Complete ranked candidate comparison table
├── metadata.json            # Dataset hash, seed, mode, environment & parameters
├── feature_importance.json  # Top features with recovered transformed names
└── report.txt               # Plain text human-readable run summary
```

---

## 🧪 Running Tests

Run the complete test suite using `pytest`:

```bash
pytest
```

Run with verbose test output:

```bash
pytest -v
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
