"""Tests for data loading and preprocessing modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.load import load_creditcard_data
from src.data.preprocess import apply_balance_strategy, prepare_train_test


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for preprocessing tests.

    Returns:
        Small dataframe with Time, V1, V2, Amount, and Class columns.
    """
    return pd.DataFrame({
        "Time": [0, 1, 2, 3, 4],
        "V1": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "V2": [0.5, 0.0, -0.5, -1.0, 1.5],
        "Amount": [100, 200, 150, 50, 300],
        "Class": [0, 0, 1, 0, 0],
    })


def test_load_creditcard_data_success(sample_dataframe, tmp_path):
    """Load data successfully from a temporary CSV file."""
    csv_path = tmp_path / "creditcard.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    df = load_creditcard_data(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ["Time", "V1", "V2", "Amount", "Class"]


def test_load_creditcard_data_file_not_found():
    """Raise FileNotFoundError when the CSV path does not exist."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_creditcard_data("nonexistent/path.csv")


def test_prepare_train_test_with_scaling(sample_dataframe):
    """Split data and return a fitted scaler when scale=True."""
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(
        sample_dataframe,
        target_column="Class",
        test_size=0.4,
        random_state=42,
        scale=True,
    )

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert scaler is not None
    assert isinstance(scaler, StandardScaler)


def test_prepare_train_test_without_scaling(sample_dataframe):
    """Return None for scaler when scale=False."""
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(
        sample_dataframe,
        target_column="Class",
        test_size=0.4,
        random_state=42,
        scale=False,
    )

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert scaler is None


def test_prepare_train_test_stratification(sample_dataframe):
    """Keep both classes present in train and test splits."""
    X_train, X_test, y_train, y_test, _ = prepare_train_test(
        sample_dataframe,
        target_column="Class",
        test_size=0.4,
        random_state=42,
    )

    assert len(np.unique(y_train)) > 0
    assert len(np.unique(y_test)) > 0


def test_prepare_train_test_excludes_time_column(sample_dataframe):
    """Exclude Time and target from feature columns."""
    X_train, X_test, y_train, y_test, _ = prepare_train_test(
        sample_dataframe,
        target_column="Class",
        test_size=0.4,
        random_state=42,
    )

    assert X_train.shape[1] == 3


def test_apply_balance_strategy_none():
    """Leave data unchanged when strategy is 'none'."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])

    X_balanced, y_balanced = apply_balance_strategy(X, y, strategy="none")

    np.testing.assert_array_equal(X_balanced, X)
    np.testing.assert_array_equal(y_balanced, y)


def test_apply_balance_strategy_oversampling():
    """Increase sample count with random oversampling."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])

    X_balanced, y_balanced = apply_balance_strategy(
        X, y, strategy="oversampling", sampling_ratio=1.0, random_state=42
    )

    assert len(X_balanced) > len(X)
    assert len(y_balanced) > len(y)


def test_apply_balance_strategy_smote():
    """Increase sample count with SMOTE."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])

    X_balanced, y_balanced = apply_balance_strategy(
        X, y, strategy="smote", sampling_ratio=1.0, random_state=42
    )

    assert len(X_balanced) > len(X)
    assert len(y_balanced) > len(y)


def test_apply_balance_strategy_invalid():
    """Raise ValueError for an unknown balancing strategy."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="Invalid strategy"):
        apply_balance_strategy(X, y, strategy="invalid_strategy")


def test_apply_balance_strategy_sampling_ratio():
    """Accept different sampling ratios for oversampling."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])

    X_balanced, y_balanced = apply_balance_strategy(
        X, y, strategy="oversampling", sampling_ratio=0.3, random_state=42
    )
    assert len(X_balanced) > len(X)

    X_balanced, y_balanced = apply_balance_strategy(
        X, y, strategy="oversampling", sampling_ratio=0.8, random_state=42
    )
    assert len(X_balanced) > len(X)
