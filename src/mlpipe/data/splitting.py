"""
Train/test splitting for MLPipe.

Implements leakage-free splitting with stratification for classification when appropriate,
and random splitting for regression.
"""

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from mlpipe.utils.logging import get_logger

logger = get_logger("splitting")


@dataclass
class SplitData:
    """Container for train/test data splits."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_size: int
    test_size: int
    is_stratified: bool


def split_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    test_size: float = 0.20,
    random_seed: int = 42,
) -> SplitData:
    """
    Split a DataFrame into train and test sets without data leakage.

    Args:
        df: Raw input DataFrame.
        target_column: Name of the target column.
        task_type: 'classification' or 'regression'.
        test_size: Fraction of samples to allocate to test set.
        random_seed: Random state for reproducibility.

    Returns:
        SplitData container with X_train, X_test, y_train, y_test.
    """
    # 1. Drop rows where target is NaN/null
    clean_df = df.dropna(subset=[target_column]).copy()
    if len(clean_df) < len(df):
        logger.info("Dropped %d rows with missing target values prior to splitting.", len(df) - len(clean_df))

    X = clean_df.drop(columns=[target_column])
    y = clean_df[target_column]

    stratify = None
    is_stratified = False

    if task_type == "classification":
        # Check if every class has at least 2 samples for stratification
        # and test set has at least as many samples as classes
        class_counts = y.value_counts()
        test_samples = int(len(clean_df) * test_size)
        if (class_counts >= 2).all() and len(class_counts) > 1 and test_samples >= len(class_counts):
            stratify = y
            is_stratified = True
        else:
            logger.warning(
                "Cannot stratify: either classes have <2 samples or test set size (%d) < classes (%d). Using random split.",
                test_samples,
                len(class_counts),
            )


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )

    # Reset indices to ensure clean indexing during transformations
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_size=len(X_train),
        test_size=len(X_test),
        is_stratified=is_stratified,
    )
