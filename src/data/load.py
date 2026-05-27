"""Dataset loading utilities."""

from pathlib import Path

import pandas as pd


def load_creditcard_data(path: str | Path) -> pd.DataFrame:
    """Load the credit card fraud dataset from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with the raw dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)
