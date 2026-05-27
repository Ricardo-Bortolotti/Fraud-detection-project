"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import classification_metrics


@pytest.fixture
def binary_predictions():
    """Sample binary labels and probabilities for metric tests.

    Returns:
        Tuple of (y_true, y_pred, y_proba).
    """
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.4, 0.3, 0.8, 0.6, 0.9])
    return y_true, y_pred, y_proba


def test_classification_metrics_basic(binary_predictions):
    """Compute all metrics when probabilities are provided."""
    y_true, y_pred, y_proba = binary_predictions

    metrics = classification_metrics(y_true, y_pred, y_proba)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["precision"], float)
    assert isinstance(metrics["recall"], float)
    assert isinstance(metrics["f1"], float)
    assert isinstance(metrics["roc_auc"], float)
    assert isinstance(metrics["pr_auc"], float)


def test_classification_metrics_without_proba(binary_predictions):
    """Omit AUC metrics when probabilities are not provided."""
    y_true, y_pred, _ = binary_predictions

    metrics = classification_metrics(y_true, y_pred, y_proba=None)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" not in metrics
    assert "pr_auc" not in metrics


def test_classification_metrics_single_class():
    """Skip AUC metrics when only one class is present."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = classification_metrics(y_true, y_pred, y_proba)

    assert "roc_auc" not in metrics
    assert "pr_auc" not in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_classification_metrics_accuracy(binary_predictions):
    """Match expected accuracy for fixed predictions."""
    y_true, y_pred, _ = binary_predictions
    metrics = classification_metrics(y_true, y_pred)

    expected_accuracy = 6 / 8
    assert abs(metrics["accuracy"] - expected_accuracy) < 0.01


def test_classification_metrics_precision(binary_predictions):
    """Match expected precision for fixed predictions."""
    y_true, y_pred, _ = binary_predictions
    metrics = classification_metrics(y_true, y_pred)

    expected_precision = 3 / 4
    assert abs(metrics["precision"] - expected_precision) < 0.01


def test_classification_metrics_recall(binary_predictions):
    """Match expected recall for fixed predictions."""
    y_true, y_pred, _ = binary_predictions
    metrics = classification_metrics(y_true, y_pred)

    expected_recall = 3 / 4
    assert abs(metrics["recall"] - expected_recall) < 0.01


def test_classification_metrics_f1(binary_predictions):
    """Match expected F1 for fixed predictions."""
    y_true, y_pred, _ = binary_predictions
    metrics = classification_metrics(y_true, y_pred)

    precision = 3 / 4
    recall = 3 / 4
    expected_f1 = 2 * (precision * recall) / (precision + recall)
    assert abs(metrics["f1"] - expected_f1) < 0.01


def test_classification_metrics_roc_auc(binary_predictions):
    """Return ROC-AUC in the valid range."""
    y_true, y_pred, y_proba = binary_predictions
    metrics = classification_metrics(y_true, y_pred, y_proba)

    assert 0 <= metrics["roc_auc"] <= 1


def test_classification_metrics_pr_auc(binary_predictions):
    """Return PR-AUC in the valid range."""
    y_true, y_pred, y_proba = binary_predictions
    metrics = classification_metrics(y_true, y_pred, y_proba)

    assert 0 <= metrics["pr_auc"] <= 1


def test_classification_metrics_zero_division():
    """Return zero precision and recall when there are no true positives."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["precision"] == 0
    assert metrics["recall"] == 0
