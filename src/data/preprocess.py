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
    """
    Separa features/alvo e divide em treino/teste com estratificação.
    Features V1–V28 e Amount são escalonadas quando scale=True.
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
    """
    Aplica estratégia de balanceamento ao conjunto de treino.

    Args:
        X_train: Features de treino
        y_train: Target de treino
        strategy: "none", "smote", ou "oversampling"
        sampling_ratio: Ratio para oversampling/SMOTE (0.0 a 1.0)
        random_state: Semente aleatória

    Returns:
        X_train_balanced, y_train_balanced

    Raises:
        ValueError: Se strategy não for válido
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
        f"Estratégia inválida: {strategy}. Use 'none', 'smote' ou 'oversampling'."
    )
