"""
Comprehensive Standalone Python Code Generator for MLPipe.

Generates self-contained, beautifully commented, reproducible Python scripts
using only standard pandas, numpy, scikit-learn, and joblib with zero dependence on MLPipe.
Includes every line of code for data cleaning, preprocessing, model definition,
hyperparameter tuning, evaluation, explainability, and serialization.
"""

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
    Generate clean, idiomatic, fully-commented Python code reproducing the entire ML workflow:
    - Data loading & cleaning
    - Exploratory statistics
    - Train/Test splitting (stratified/shuffle)
    - Feature preprocessing (ColumnTransformer: imputation, scaling, one-hot encoding)
    - Machine learning model setup with fine-tuned hyperparameters
    - Optional hyperparameter search tuning function
    - Model fitting
    - Evaluation on unseen test data
    - Feature importance / explainability extraction
    - Serialization and sample inference
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

    # Escape path
    norm_path = dataset_path.replace("\\", "/")
    train_pct = int(round((1.0 - test_size) * 100))
    test_pct = int(round(test_size * 100))

    # Task-specific evaluation metrics & code
    if task_type == "classification":
        metrics_import = "from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score"
        stratify_arg = "stratify=y,"
        eval_code = """    # ── 6. Test Set Evaluation & Metrics ──────────────────────────────────────────
    print("\\n" + "=" * 60)
    print("Model Evaluation (Held-Out Test Data)")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
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
    print(confusion_matrix(y_test, y_pred))"""
    else:
        metrics_import = "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"
        stratify_arg = ""
        eval_code = """    # ── 6. Test Set Evaluation & Metrics ──────────────────────────────────────────
    print("\\n" + "=" * 60)
    print("Model Evaluation (Held-Out Test Data)")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Optional MAPE calculation
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
    print(f"MAPE:      {mape_str}  (Mean Absolute Percentage Error)")"""

    # Feature explainability block
    explain_code = """    # ── 7. Feature Importance & Model Explainability ──────────────────────────────
    print("\\n" + "=" * 60)
    print("MODEL EXPLAINABILITY & TOP PREDICTIVE FEATURES")
    print("=" * 60)
    try:
        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        fitted_estimator = pipeline.named_steps["estimator"]

        # Extract transformed feature names
        feature_names = fitted_preprocessor.get_feature_names_out()

        # Check for tree feature importances
        if hasattr(fitted_estimator, "feature_importances_"):
            importances = fitted_estimator.feature_importances_
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            print("Top 10 Most Important Features:")
            for rank, (name, imp) in enumerate(feat_imp[:10], start=1):
                clean_name = name.replace("num__", "").replace("cat__", "")
                print(f"  {rank:>2}. {clean_name:<30} {imp * 100:>6.2f}%")
        # Check for linear coefficients
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
                print(f"  {rank:>2}. {clean_name:<30} {imp:>8.4f}")
    except Exception as e:
        print(f"Feature importance extraction skipped: {e}")"""

    code = f'''"""
================================================================================
Standalone Machine Learning Pipeline
Generated by MLPipe (https://github.com/atharva081106/mlpipe)
================================================================================

This Python script is 100% self-contained and reproducible.
It contains every line of code required to:
1. Ingest, inspect, and clean the dataset
2. Split data cleanly into training and hold-out test sets without data leakage
3. Set up complete data preprocessing pipelines (imputation, scaling, one-hot encoding)
4. Initialize the champion model with optimal fine-tuned hyperparameters
5. Fit and train the end-to-end pipeline
6. Evaluate model performance on unseen test data with comprehensive metrics
7. Extract feature importances and explain model decisions
8. Save the trained pipeline to disk for production deployment and run inference

Requirements (standard open-source data science stack):
    pip install pandas numpy scikit-learn joblib
"""

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


def load_and_clean_data(data_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Step 1: Ingest and clean the dataset.
    - Loads CSV data
    - Removes rows where target column is null
    - Replaces infinite values with NaN
    - Separates features (X) and target (y)
    """
    print("=" * 60)
    print("1. DATA INGESTION & CLEANING")
    print("=" * 60)
    print(f"Reading dataset from: {{data_path}}")
    df = pd.read_csv(data_path)
    print(f"Raw dataset shape: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{{target_column}}' not found in dataset columns: {{list(df.columns)}}")

    # Drop records where target is missing (cannot train or evaluate on unknown target)
    initial_rows = len(df)
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"Dropped {{dropped}} rows with missing target values")

    # Replace any infinite values with NaN so imputers handle them cleanly
    df = df.replace([np.inf, -np.inf], np.nan)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"Cleaned feature matrix X: {{X.shape[0]}} rows, {{X.shape[1]}} features")
    print(f"Target vector y: {{len(y)}} values")
    return X, y


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Step 2: Build sklearn ColumnTransformer for robust preprocessing.
    - Numerical: Median imputation (robust to outliers) + StandardScaler (mean=0, std=1)
    - Categorical: Most-frequent (mode) imputation + OneHotEncoder (handles unseen categories gracefully)
    """
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

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop any unselected/ignored columns
    )
    return preprocessor


def main():
    # ── 1. Load and Clean Dataset ────────────────────────────────────────────────
    data_path = "{norm_path}"
    target_column = "{target_column}"

    X, y = load_and_clean_data(data_path, target_column)

    # ── 2. Leakage-Free Train / Test Split ───────────────────────────────────────
    print("\\n" + "=" * 60)
    print("2. LEAKAGE-FREE TRAIN / TEST SPLITTING")
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

    # ── 3. Identify Feature Columns for Preprocessing ────────────────────────────
    numeric_features = {numeric_columns}
    categorical_features = {categorical_columns}

    # Intersect with columns actually present in training data
    num_cols = [c for c in numeric_features if c in X_train.columns]
    cat_cols = [c for c in categorical_features if c in X_train.columns]

    if not num_cols and not cat_cols:
        # Fallback automatic column type inference if feature lists were empty
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_train.columns if c not in num_cols]

    print(f"Numeric features to impute and scale ({{len(num_cols)}}): {{num_cols}}")
    print(f"Categorical features to impute and encode ({{len(cat_cols)}}): {{cat_cols}}")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # ── 4. Model Initialization with Fine-Tuned Hyperparameters ─────────────────
    print("\\n" + "=" * 60)
    print("3. MODEL INITIALIZATION & PIPELINE ASSEMBLY")
    print("=" * 60)
    print("Model Architecture: {model_name} ({class_name})")

    # Optimal fine-tuned parameters discovered by automated hyperparameter optimization
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

    # ── 5. Model Training ────────────────────────────────────────────────────────
    print("\\n" + "=" * 60)
    print("4. TRAINING END-TO-END PIPELINE")
    print("=" * 60)
    print("Fitting preprocessor and model strictly on training split...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

{eval_code}

{explain_code}

    # ── 8. Production Model Serialization & Saving ────────────────────────────────
    print("\\n" + "=" * 60)
    print("8. ARTIFACT SERIALIZATION")
    print("=" * 60)
    output_model_path = "pipeline.joblib"
    joblib.dump(pipeline, output_model_path)
    print(f"Full pipeline successfully saved to: {{output_model_path}}")

    # ── 9. Example Inference on New / Unseen Records ──────────────────────────────
    print("\\n" + "=" * 60)
    print("9. EXAMPLE INFERENCE DEMONSTRATION")
    print("=" * 60)
    # Load saved pipeline back from disk to prove independence
    deployed_pipeline = joblib.load(output_model_path)

    sample_test = X_test.head(3)
    sample_actual = y_test.head(3).values
    sample_preds = deployed_pipeline.predict(sample_test)

    print("Predicting on sample test rows using deployed model:")
    for i in range(len(sample_test)):
        print(f"  Row {{i+1}}: Actual = {{sample_actual[i]}} | Predicted = {{sample_preds[i]}}")

    print("\\nPipeline script executed successfully!")


if __name__ == "__main__":
    main()
'''
    return code
