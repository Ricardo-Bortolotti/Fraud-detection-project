"""Logistic regression estimator factory."""

from typing import Any

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(
    max_iter: int = 1000,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    tol: float = 1e-4,
    **kwargs: Any,
) -> LogisticRegression:
    """Build a logistic regression classifier for fraud detection.

    Args:
        max_iter: Maximum iterations for the solver.
        class_weight: Class weights or ``"balanced"``.
        random_state: Random seed.
        C: Inverse regularization strength.
        penalty: Regularization norm (``"l1"``, ``"l2"``, or ``"elasticnet"``).
        solver: Optimization algorithm.
        tol: Convergence tolerance.
        **kwargs: Additional arguments passed to ``LogisticRegression``.

    Returns:
        Configured ``LogisticRegression`` instance.
    """
    return LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        C=C,
        penalty=penalty,
        solver=solver,
        tol=tol,
        **kwargs,
    )
