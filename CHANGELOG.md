# Changelog

All notable changes to MLPipe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-07

### Added
- Core `Pipeline` class supporting automated ML for tabular datasets.
- Automatic task detection (Binary Classification, Multiclass Classification, Regression).
- Robust CSV ingestion with file validation and SHA-256 dataset hashing.
- Dataset profiling reporting rows, columns, memory, types, missing values, duplicates, and distributions.
- Pre-training data validation with actionable fatal errors and warnings.
- Leakage-free train/test splitting (stratified for classification, random for regression).
- Automated sklearn `ColumnTransformer` preprocessing with median imputation, scaling, and one-hot encoding.
- Multi-model candidate evaluation:
  - Classification: Logistic Regression, Random Forest, HistGradientBoosting, Decision Tree, K-Nearest Neighbors.
  - Regression: Ridge, Random Forest Regressor, HistGradientBoosting Regressor, Decision Tree Regressor.
- Cross-validation and hyperparameter tuning with `RandomizedSearchCV` across `fast`, `balanced`, and `thorough` modes.
- Best model selection based strictly on CV score on the training set (no test-set selection).
- Unbiased final evaluation on held-out test set with classification & regression metrics.
- Feature importance extraction for tree-based and linear models with recovered post-preprocessing feature names.
- Complete artifact management: `model.joblib`, `pipeline.joblib`, `metrics.json`, `leaderboard.json`, `metadata.json`, `feature_importance.json`, and `report.txt`.
- Rich-powered Terminal CLI (`mlpipe`) with subcommands:
  - `mlpipe train`
  - `mlpipe profile`
  - `mlpipe validate`
  - `mlpipe predict`
  - `mlpipe inspect`
  - `mlpipe version`
- Reusable pipeline persistence (`save` / `load`) and inference (`predict`).
- Synthetic deterministic demo datasets: `customer_churn.csv` and `house_prices.csv`.
- Comprehensive test suite covering ingestion, profiling, validation, preprocessing, training, tuning, evaluation, leakage prevention, serialization, and CLI.
