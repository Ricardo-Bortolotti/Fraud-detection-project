"""XGBoost estimator factory."""

from typing import Any

from xgboost import XGBClassifier


def build_xgboost(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
    **kwargs: Any,
) -> XGBClassifier:
    """Build an XGBoost classifier for fraud detection.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        subsample: Row subsample ratio per tree.
        colsample_bytree: Column subsample ratio per tree.
        min_child_weight: Minimum sum of instance weight in a child.
        gamma: Minimum loss reduction for a split.
        reg_alpha: L1 regularization on weights.
        reg_lambda: L2 regularization on weights.
        scale_pos_weight: Weight for the positive class.
        n_jobs: Parallel jobs; ``-1`` uses all cores.
        random_state: Random seed.
        **kwargs: Additional arguments passed to ``XGBClassifier``.

    Returns:
        Configured ``XGBClassifier`` with binary logistic objective.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        objective="binary:logistic",
        eval_metric="auc",
        **kwargs,
    )
