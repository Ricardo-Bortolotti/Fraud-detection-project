from typing import Tuple

import numpy as np
import pandas as pd
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
