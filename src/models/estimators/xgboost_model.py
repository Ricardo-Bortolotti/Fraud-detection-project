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
    """XGBoost para classificação de fraude."""
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
