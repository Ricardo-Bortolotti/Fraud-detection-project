from typing import Any

import mlflow
import numpy as np
from sklearn.base import ClassifierMixin

from src.evaluation.metrics import classification_metrics


def train_and_log(
    model: ClassifierMixin,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Treina o modelo, calcula métricas e registra no MLflow."""
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

    return metrics
