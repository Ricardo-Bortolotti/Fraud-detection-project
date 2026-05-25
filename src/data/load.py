from pathlib import Path

import pandas as pd


def load_creditcard_data(path: str | Path) -> pd.DataFrame:
    """Carrega o dataset de fraude em cartão de crédito."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    return pd.read_csv(path)
