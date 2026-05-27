"""MLflow training and logging utilities."""

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import classification_metrics


def train_and_log(
    model: ClassifierMixin,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    params: dict[str, Any] | None = None,
    scaler: StandardScaler | None = None,
) -> dict[str, float]:
    """Train a classifier, evaluate on the test set, and log to MLflow.

    Args:
        model: Scikit-learn compatible classifier.
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        model_name: Run name and MLflow model tag.
        params: Hyperparameters and run metadata to log.
        scaler: Optional fitted scaler persisted as an artifact when provided.

    Returns:
        Test-set metrics from :func:`classification_metrics`.
    """
    params = params or {}

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, y_pred, y_proba)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.set_tag("model_type", model_name)

        if scaler is not None:
            scaler_path = Path("models/champion/model/model/scaler.joblib")
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact("scaler.joblib", artifact_path="model")

    return metrics
