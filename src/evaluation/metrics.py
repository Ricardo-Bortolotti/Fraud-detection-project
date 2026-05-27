"""Classification metrics for fraud detection."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute binary classification metrics for fraud detection.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted class labels.
        y_proba: Predicted positive-class probabilities. When provided and
            both classes are present in ``y_true``, ROC-AUC and PR-AUC are
            included.

    Returns:
        Dictionary with at least ``accuracy``, ``precision``, ``recall``, and
        ``f1``. May also include ``roc_auc`` and ``pr_auc``.
    """
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return metrics
