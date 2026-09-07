"""
Comprehensive Standalone Python Code & Jupyter Notebook Generator for MLPipe.

Generates self-contained, beautifully commented, reproducible Python scripts (.py)
and Jupyter Notebooks (.ipynb) using only standard pandas, numpy, scikit-learn,
matplotlib, and joblib with ZERO dependence on MLPipe.

Includes every line of code for data cleaning, preprocessing, model definition,
hyperparameter tuning, evaluation, explainability, visual plotting, and serialization.
Fully structured with '# %%' cells ready to run interactively in VS Code or Jupyter.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pprint


MODEL_IMPORTS = {
    "Logistic Regression": ("sklearn.linear_model", "LogisticRegression"),
    "Random Forest": {
        "classification": ("sklearn.ensemble", "RandomForestClassifier"),
        "regression": ("sklearn.ensemble", "RandomForestRegressor"),
    },
    "HistGradientBoosting": {
        "classification": ("sklearn.ensemble", "HistGradientBoostingClassifier"),
        "regression": ("sklearn.ensemble", "HistGradientBoostingRegressor"),
    },
    "Decision Tree": {
        "classification": ("sklearn.tree", "DecisionTreeClassifier"),
        "regression": ("sklearn.tree", "DecisionTreeRegressor"),
    },
    "K-Nearest Neighbors": {
        "classification": ("sklearn.neighbors", "KNeighborsClassifier"),
        "regression": ("sklearn.neighbors", "KNeighborsRegressor"),
    },
    "Ridge Regression": ("sklearn.linear_model", "Ridge"),
    "Linear Regression": ("sklearn.linear_model", "LinearRegression"),
    "Extra Trees": {
        "classification": ("sklearn.ensemble", "ExtraTreesClassifier"),
        "regression": ("sklearn.ensemble", "ExtraTreesRegressor"),
    },
}


def _get_model_import_and_class(model_name: str, task_type: str) -> tuple[str, str]:
    """Resolve module path and class name for a given model."""
    for key, spec in MODEL_IMPORTS.items():
        if key.lower() in model_name.lower():
            if isinstance(spec, dict):
                return spec[task_type]
            return spec

    # Default fallback
    if task_type == "classification":
        return ("sklearn.ensemble", "RandomForestClassifier")
    return ("sklearn.ensemble", "RandomForestRegressor")


def generate_standalone_code(
    dataset_path: str,
    target_column: str,
    task_type: str,
    model_name: str,
    estimator_params: Dict[str, Any],
    numeric_columns: List[str],
    categorical_columns: List[str],
    test_size: float = 0.20,
    random_seed: int = 42,
    include_tuning_block: bool = True,
) -> str:
    """
    Generate clean, idiomatic, fully-commented Python code reproducing the entire ML workflow.
    Structured with '# %%' interactive cells ready to run cell-by-cell in VS Code or PyCharm,
    and runs top-to-bottom as a standard Python script.
    """
    mod_path, class_name = _get_model_import_and_class(model_name, task_type)

    # Filter estimator parameters to relevant, serializable kwargs
    clean_params = {}
    for k, v in estimator_params.items():
        if k.startswith("_") or callable(v):
            continue
        if isinstance(v, (int, float, str, bool, list)) or v is None:
            clean_params[k] = v

    formatted_params = pprint.pformat(clean_params, indent=8, width=80)

    # Clean path & filename for fallback lookup
    norm_path = dataset_path.replace("\\", "/")
    filename = Path(norm_path).name
    train_pct = int(round((1.0 - test_size) * 100))
    test_pct = int(round(test_size * 100))

    if task_type == "classification":
        metrics_import = "from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score"
        stratify_arg = "stratify=y,"
        eval_metrics_code = """    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:          {acc:.4f}  (Percentage of correct predictions)")
    print(f"Balanced Accuracy: {bal_acc:.4f}  (Accounts for class imbalance)")
    print(f"Precision:         {prec:.4f}  (Weighted precision across classes)")
    print(f"Recall:            {rec:.4f}  (Weighted recall across classes)")
    print(f"F1 Score:          {f1:.4f}  (Harmonic mean of precision and recall)")

    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                roc = roc_auc_score(y_test, y_proba[:, 1])
                print(f"ROC-AUC Score:     {roc:.4f}  (Area Under ROC Curve: 0.5=random, 1.0=perfect)")
            else:
                roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                print(f"Multiclass ROC-AUC:{roc:.4f}  (One-vs-Rest weighted ROC-AUC)")
        except Exception as e:
            print(f"(ROC-AUC calculation skipped: {e})")

    print("\\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Visual Plot: Confusion Matrix (renders in VS Code Interactive Window & Jupyter)
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        plt.title(f"Confusion Matrix — {class_name}")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass"""
    else:
        metrics_import = "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"
        stratify_arg = ""
        eval_metrics_code = """    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    non_zero = y_test != 0
    if np.any(non_zero):
        mape = np.mean(np.abs((y_test[non_zero] - y_pred[non_zero]) / y_test[non_zero])) * 100
        mape_str = f"{mape:.2f}%"
    else:
        mape_str = "N/A"

    print(f"R² Score:  {r2:.4f}  (1.0 = perfect fit, 0.0 = baseline average)")
    print(f"MAE:       {mae:.4f}  (Mean Absolute Error in target units)")
    print(f"MSE:       {mse:.4f}  (Mean Squared Error)")
    print(f"RMSE:      {rmse:.4f}  (Root Mean Squared Error: standard deviation of residuals)")
    print(f"MAPE:      {mape_str}  (Mean Absolute Percentage Error)")

    # Visual Plot: Actual vs. Predicted (renders in VS Code Interactive Window & Jupyter)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", color="#2b5c8f")
        min_v = float(min(np.min(y_test), np.min(y_pred)))
        max_v = float(max(np.max(y_test), np.max(y_pred)))
        plt.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Ideal 1:1 Fit")
        plt.xlabel("Actual Ground Truth")
        plt.ylabel("Model Prediction")
        plt.title(f"Actual vs. Predicted — {class_name} (R² = {r2:.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass"""

    code = f'''# %% [markdown]
# # Machine Learning Pipeline
# **Generated by MLPipe** (https://github.com/atharva081106/mlpipe)
# 
# This script is 100% self-contained and reproducible.
# - **Target Column**: `{target_column}`
# - **Problem Type**: `{task_type.capitalize()}`
# - **Winning Model**: `{model_name}` (`{class_name}`)
# 
# ### VS Code & Jupyter Ready
# Each section is demarcated with `# %%` cell markers.
# You can click **"Run Cell"** in VS Code to execute step-by-step, or run directly via terminal:
# ```bash
# python reproduce_pipeline.py
# ```

# %% [markdown]
# ## 1. Setup & Library Imports
# Imports standard, battle-tested open-source data science packages: `pandas`, `numpy`, `scikit-learn`, `joblib`.

# %%
from pathlib import Path
import pprint
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
{metrics_import}
from {mod_path} import {class_name}
import joblib


# %% [markdown]
# ## 2. Data Ingestion & Cleaning Helpers
# Loads CSV data, handles missing targets, replaces inf values, and isolates features $X$ and target $y$.

# %%
def resolve_data_path(raw_path: str) -> Path:
    """Robust path resolver: checks exact path, then current working directory, then script directory."""
    p = Path(raw_path)
    if p.exists():
        return p
    # Fallback to current working directory
    p_cwd = Path.cwd() / "{filename}"
    if p_cwd.exists():
        return p_cwd
    # Fallback to directory of current file
    try:
        p_file = Path(__file__).resolve().parent / "{filename}"
        if p_file.exists():
            return p_file
    except NameError:
        pass
    return p


def load_and_clean_data(data_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset, clean invalid records, and separate features X and target y."""
    print("=" * 60)
    print("1. DATA INGESTION & CLEANING")
    print("=" * 60)
    print(f"Reading dataset from: {{data_path}}")
    df = pd.read_csv(data_path)
    print(f"Raw dataset shape: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{{target_col}}' not found in dataset columns: {{list(df.columns)}}")

    # Drop rows where target is missing
    initial_rows = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"Dropped {{dropped}} rows with missing target values")

    # Replace inf with NaN for clean imputation
    df = df.replace([np.inf, -np.inf], np.nan)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Cleaned feature matrix X: {{X.shape[0]}} rows, {{X.shape[1]}} features")
    print(f"Target vector y: {{len(y)}} values")
    return X, y


# %% [markdown]
# ## 3. Feature Preprocessing Pipeline
# Constructs an automated `ColumnTransformer`:
# - Numerical features: Median Imputation + Standard Scaling (mean=0, std=1)
# - Categorical features: Most-Frequent Imputation + One-Hot Encoding (ignoring unseen categories)

# %%
def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for robust preprocessing."""
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# %% [markdown]
# ## 4. Main Execution Pipeline

# %%
def main():
    # ── 1. Ingest Dataset ─────────────────────────────────────────────────────────
    dataset_file = resolve_data_path("{norm_path}")
    target_col = "{target_column}"

    X, y = load_and_clean_data(dataset_file, target_col)

    # ── 2. Leakage-Free Train / Test Split ───────────────────────────────────────
    print("\\n" + "=" * 60)
    print("2. LEAKAGE-FREE TRAIN / TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size={test_size},
        random_state={random_seed},
        {stratify_arg}
    )
    print(f"Training set: {{len(X_train):,}} samples ({train_pct}%)")
    print(f"Testing set:  {{len(X_test):,}} samples ({test_pct}%)")

    # ── 3. Assemble Feature Preprocessing Pipeline ───────────────────────────────
    numeric_features = {numeric_columns}
    categorical_features = {categorical_columns}

    num_cols = [c for c in numeric_features if c in X_train.columns]
    cat_cols = [c for c in categorical_features if c in X_train.columns]

    if not num_cols and not cat_cols:
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_train.columns if c not in num_cols]

    print(f"\\nNumeric features to scale ({{len(num_cols)}}): {{num_cols}}")
    print(f"Categorical features to encode ({{len(cat_cols)}}): {{cat_cols}}")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # ── 4. Initialize Estimator with Optimal Hyperparameters ─────────────────────
    print("\\n" + "=" * 60)
    print("3. MODEL INITIALIZATION & PIPELINE ASSEMBLY")
    print("=" * 60)
    print("Model Architecture: {model_name} ({class_name})")

    model_params = {formatted_params}

    print("Configured Model Hyperparameters:")
    for param_name, param_val in model_params.items():
        print(f"  • {{param_name}} = {{param_val}}")

    estimator = {class_name}(**model_params)

    # Assemble full end-to-end scikit-learn Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])

    # ── 5. Train Model Pipeline ──────────────────────────────────────────────────
    print("\\n" + "=" * 60)
    print("4. TRAINING END-TO-END PIPELINE")
    print("=" * 60)
    print("Fitting preprocessor and model strictly on training split...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # ── 6. Test Set Evaluation & Metrics ──────────────────────────────────────────
    print("\\n" + "=" * 60)
    print("5. MODEL EVALUATION (HELD-OUT TEST DATA)")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)
{eval_metrics_code}

    # ── 7. Feature Importance & Model Explainability ──────────────────────────────
    print("\\n" + "=" * 60)
    print("6. MODEL EXPLAINABILITY & TOP PREDICTIVE FEATURES")
    print("=" * 60)
    feat_imp = []
    try:
        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        fitted_estimator = pipeline.named_steps["estimator"]
        feature_names = fitted_preprocessor.get_feature_names_out()

        if hasattr(fitted_estimator, "feature_importances_"):
            importances = fitted_estimator.feature_importances_
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            print("Top 10 Most Important Features:")
            for rank, (name, imp) in enumerate(feat_imp[:10], start=1):
                clean_name = name.replace("num__", "").replace("cat__", "")
                print(f"  {{rank:>2}}. {{clean_name:<30}} {{imp * 100:>6.2f}}%")
        elif hasattr(fitted_estimator, "coef_"):
            coefs = fitted_estimator.coef_
            if coefs.ndim > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            else:
                coefs = np.abs(coefs)
            feat_imp = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)
            print("Top 10 Most Influential Feature Coefficients (Absolute Value):")
            for rank, (name, imp) in enumerate(feat_imp[:10], start=1):
                clean_name = name.replace("num__", "").replace("cat__", "")
                print(f"  {{rank:>2}}. {{clean_name:<30}} {{imp:>8.4f}}")

        # Visual Plot: Horizontal Bar Chart for Feature Importances
        try:
            import matplotlib.pyplot as plt
            if feat_imp:
                top_feats = feat_imp[:10][::-1]
                p_names = [x[0].replace("num__", "").replace("cat__", "") for x in top_feats]
                p_scores = [x[1] for x in top_feats]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(p_names, p_scores, color="#16a085")
                ax.set_xlabel("Relative Importance / Influence")
                ax.set_title("Top 10 Features Driving Model Decisions")
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                plt.show()
        except Exception:
            pass

    except Exception as e:
        print(f"Feature importance extraction skipped: {{e}}")

    # ── 8. Production Model Serialization & Saving ────────────────────────────────
    print("\\n" + "=" * 60)
    print("7. ARTIFACT SERIALIZATION")
    print("=" * 60)
    output_model_path = "pipeline.joblib"
    joblib.dump(pipeline, output_model_path)
    print(f"Full pipeline successfully saved to: {{output_model_path}}")

    # ── 9. Example Inference on New / Unseen Records ──────────────────────────────
    print("\\n" + "=" * 60)
    print("8. EXAMPLE INFERENCE DEMONSTRATION")
    print("=" * 60)
    deployed_pipeline = joblib.load(output_model_path)
    sample_test = X_test.head(3)
    sample_actual = y_test.head(3).values
    sample_preds = deployed_pipeline.predict(sample_test)

    print("Predicting on sample test rows using deployed model:")
    for i in range(len(sample_test)):
        print(f"  Row {{i+1}}: Actual = {{sample_actual[i]}} | Predicted = {{sample_preds[i]}}")

    print("\\n[SUCCESS] Pipeline executed successfully!")
    return pipeline, X_train, X_test, y_train, y_test


# %%
if __name__ == "__main__":
    main()
'''
    return code


def generate_standalone_notebook(
    dataset_path: str,
    target_column: str,
    task_type: str,
    model_name: str,
    estimator_params: Dict[str, Any],
    numeric_columns: List[str],
    categorical_columns: List[str],
    test_size: float = 0.20,
    random_seed: int = 42,
) -> str:
    """
    Generate a complete, self-contained Jupyter Notebook (.ipynb)
    ready to run cell-by-cell in VS Code, JupyterLab, or Google Colab.
    Includes Markdown explanations, scikit-learn pipeline, model evaluation,
    and inline matplotlib visualizations.
    """
    mod_path, class_name = _get_model_import_and_class(model_name, task_type)

    clean_params = {}
    for k, v in estimator_params.items():
        if k.startswith("_") or callable(v):
            continue
        if isinstance(v, (int, float, str, bool, list)) or v is None:
            clean_params[k] = v

    formatted_params = pprint.pformat(clean_params, indent=4, width=80)
    norm_path = dataset_path.replace("\\", "/")
    filename = Path(norm_path).name

    if task_type == "classification":
        metrics_import = "from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay"
        stratify_arg = "stratify=y,"
        eval_code_nb = f"""y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"Accuracy:          {{acc:.4f}}")
print(f"Balanced Accuracy: {{bal_acc:.4f}}")
print(f"Precision:         {{prec:.4f}}")
print(f"Recall:            {{rec:.4f}}")
print(f"F1 Score:          {{f1:.4f}}")

if hasattr(pipeline, "predict_proba"):
    try:
        y_proba = pipeline.predict_proba(X_test)
        if y_proba.shape[1] == 2:
            roc = roc_auc_score(y_test, y_proba[:, 1])
            print(f"ROC-AUC Score:     {{roc:.4f}}")
        else:
            roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            print(f"Multiclass ROC-AUC:{{roc:.4f}}")
    except Exception as e:
        print(f"ROC-AUC calculation skipped: {{e}}")

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
plt.title("Confusion Matrix — {class_name}")
plt.tight_layout()
plt.show()"""
    else:
        metrics_import = "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"
        stratify_arg = ""
        eval_code_nb = f"""y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R² Score:  {{r2:.4f}}  (1.0 = perfect prediction, 0.0 = baseline average)")
print(f"MAE:       {{mae:.4f}}  (Mean Absolute Error)")
print(f"MSE:       {{mse:.4f}}  (Mean Squared Error)")
print(f"RMSE:      {{rmse:.4f}}  (Root Mean Squared Error)")

# Plot Actual vs. Predicted Scatter
fig, ax = plt.subplots(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", color="#2b5c8f")
min_v = float(min(np.min(y_test), np.min(y_pred)))
max_v = float(max(np.max(y_test), np.max(y_pred)))
plt.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Ideal 1:1 Fit")
plt.xlabel("Actual Ground Truth")
plt.ylabel("Model Prediction")
plt.title(f"Actual vs. Predicted — {class_name} (R² = {{r2:.4f}})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Machine Learning Pipeline — `{target_column}`\n",
                f"**Generated by MLPipe** — Ready to execute interactively in Jupyter or VS Code.\n\n",
                f"- **Target**: `{target_column}`\n",
                f"- **Task Type**: `{task_type.capitalize()}`\n",
                f"- **Winning Algorithm**: `{model_name}` (`{class_name}`)\n",
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Imports & Environment Setup\n",
                "Load standard open-source packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `joblib`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
                f"{metrics_import}\n",
                f"from {mod_path} import {class_name}\n",
                "import joblib\n",
                "\n",
                "# Set plot styling\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Ingest and Inspect Dataset\n",
                "Load CSV file and inspect shape, columns, and initial rows."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"data_path = Path('{norm_path}')\n",
                f"if not data_path.exists():\n",
                f"    data_path = Path('{filename}')  # Fallback to local working directory\n",
                "\n",
                f"df = pd.read_csv(data_path)\n",
                "print(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Cleaning & Feature Separation\n",
                f"Isolate feature matrix $X$ and target vector $y$ (`{target_column}`). Clean nulls and infinite values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"target_col = '{target_column}'\n",
                "# Drop records with missing target values\n",
                "df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)\n",
                "df_clean = df_clean.replace([np.inf, -np.inf], np.nan)\n",
                "\n",
                "X = df_clean.drop(columns=[target_col])\n",
                "y = df_clean[target_col]\n",
                "print(f'Clean feature matrix X: {X.shape}')\n",
                "print(f'Clean target vector y: {y.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Leakage-Free Train / Test Split\n",
                f"Split into 80% Training and 20% Hold-out Test sets to ensure unbiased evaluation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"X_train, X_test, y_train, y_test = train_test_split(\n",
                f"    X, y, test_size={test_size}, random_state={random_seed}, {stratify_arg}\n",
                f")\n",
                "print(f'Training split: {X_train.shape[0]} samples')\n",
                "print(f'Testing split:  {X_test.shape[0]} samples')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Feature Preprocessing Pipeline\n",
                "Build a scikit-learn `ColumnTransformer` with median imputation + standard scaling for numeric features, and mode imputation + one-hot encoding for categorical features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"numeric_features = [c for c in {numeric_columns} if c in X_train.columns]\n",
                f"categorical_features = [c for c in {categorical_columns} if c in X_train.columns]\n",
                "\n",
                "transformers = []\n",
                "if numeric_features:\n",
                "    num_pipe = Pipeline([\n",
                "        ('imputer', SimpleImputer(strategy='median')),\n",
                "        ('scaler', StandardScaler())\n",
                "    ])\n",
                "    transformers.append(('num', num_pipe, numeric_features))\n",
                "\n",
                "if categorical_features:\n",
                "    cat_pipe = Pipeline([\n",
                "        ('imputer', SimpleImputer(strategy='most_frequent')),\n",
                "        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n",
                "    ])\n",
                "    transformers.append(('cat', cat_pipe, categorical_features))\n",
                "\n",
                "preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')\n",
                "print(f'Preprocessor configured with {len(numeric_features)} numeric and {len(categorical_features)} categorical features.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model Definition & Pipeline Assembly\n",
                f"Initialize the `{model_name}` estimator using optimal fine-tuned hyperparameters discovered by MLPipe."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"model_params = {formatted_params}\n",
                f"estimator = {class_name}(**model_params)\n",
                "\n",
                "pipeline = Pipeline(steps=[\n",
                "    ('preprocessor', preprocessor),\n",
                "    ('estimator', estimator)\n",
                "])\n",
                "pipeline"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Model Training\n",
                "Train the entire pipeline strictly on the training set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pipeline.fit(X_train, y_train)\n",
                "print('Training complete!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Test Set Evaluation & Metrics\n",
                "Evaluate model predictions against held-out ground truth."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [eval_code_nb]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Feature Importance & Model Explainability\n",
                "Visualize the top features driving the model's predictions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fitted_preprocessor = pipeline.named_steps['preprocessor']\n",
                "fitted_estimator = pipeline.named_steps['estimator']\n",
                "feature_names = fitted_preprocessor.get_feature_names_out()\n",
                "\n",
                "feat_imp = []\n",
                "if hasattr(fitted_estimator, 'feature_importances_'):\n",
                "    feat_imp = sorted(zip(feature_names, fitted_estimator.feature_importances_), key=lambda x: x[1], reverse=True)\n",
                "elif hasattr(fitted_estimator, 'coef_'):\n",
                "    coefs = np.abs(fitted_estimator.coef_)\n",
                "    if coefs.ndim > 1:\n",
                "        coefs = np.mean(coefs, axis=0)\n",
                "    feat_imp = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)\n",
                "\n",
                "if feat_imp:\n",
                "    top_feats = feat_imp[:10][::-1]\n",
                "    p_names = [x[0].replace('num__', '').replace('cat__', '') for x in top_feats]\n",
                "    p_scores = [x[1] for x in top_feats]\n",
                "\n",
                "    fig, ax = plt.subplots(figsize=(8, 5))\n",
                "    ax.barh(p_names, p_scores, color='#16a085')\n",
                "    ax.set_xlabel('Relative Importance / Weight')\n",
                "    ax.set_title('Top 10 Most Influential Features')\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Model Serialization & Production Inference\n",
                "Save the trained pipeline to disk as `pipeline.joblib` and perform sample inference."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_output_file = 'pipeline.joblib'\n",
                "joblib.dump(pipeline, model_output_file)\n",
                "print(f'Model saved to {model_output_file}')\n",
                "\n",
                "# Reload and run sample prediction\n",
                "loaded_model = joblib.load(model_output_file)\n",
                "sample_rows = X_test.head(3)\n",
                "sample_preds = loaded_model.predict(sample_rows)\n",
                "\n",
                "inference_df = sample_rows.copy()\n",
                "inference_df['Actual'] = y_test.head(3).values\n",
                "inference_df['Predicted'] = sample_preds\n",
                "inference_df"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return json.dumps(notebook, indent=2)
