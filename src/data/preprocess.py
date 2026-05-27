"""Training data preparation and class balancing."""

from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_train_test(
    df: pd.DataFrame,
    target_column: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """Split features and target into stratified train and test sets.

    Features V1–V28 and Amount are standardized when ``scale=True``.
    The ``Time`` column is excluded from features.

    Args:
        df: Input dataframe.
        target_column: Name of the binary target column.
        test_size: Fraction of rows held out for testing.
        random_state: Random seed for reproducibility.
        scale: Whether to fit ``StandardScaler`` on training features.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler). ``scaler`` is
        ``None`` when ``scale=False``.
    """
    feature_cols = [c for c in df.columns if c not in (target_column, "Time")]
    X = df[feature_cols].copy()
    y = df[target_column].values

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, scaler


def apply_balance_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "none",
    sampling_ratio: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a class-balancing strategy to the training set.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        strategy: One of ``"none"``, ``"smote"``, or ``"oversampling"``.
        sampling_ratio: Target ratio for minority oversampling (0.0 to 1.0).
        random_state: Random seed for resampling.

    Returns:
        Resampled (X_train, y_train). Unchanged when ``strategy`` is ``"none"``.

    Raises:
        ValueError: If ``strategy`` is not recognized.
    """
    if strategy == "none":
        return X_train, y_train

    if strategy == "smote":
        smote = SMOTE(sampling_strategy=sampling_ratio, random_state=random_state)
        return smote.fit_resample(X_train, y_train)

    if strategy == "oversampling":
        oversampler = RandomOverSampler(
            sampling_strategy=sampling_ratio, random_state=random_state
        )
        return oversampler.fit_resample(X_train, y_train)

    raise ValueError(
        f"Invalid strategy: {strategy}. Use 'none', 'smote', or 'oversampling'."
    )
