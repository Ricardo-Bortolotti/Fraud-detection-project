"""Random forest estimator factory."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | float = "sqrt",
    class_weight: str | dict | None = "balanced",
    n_jobs: int = -1,
    random_state: int = 42,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Build a random forest classifier for fraud detection.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth; ``None`` means unlimited.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples per leaf.
        max_features: Number of features considered per split.
        class_weight: Class weights or ``"balanced"``.
        n_jobs: Parallel jobs; ``-1`` uses all cores.
        random_state: Random seed.
        **kwargs: Additional arguments passed to ``RandomForestClassifier``.

    Returns:
        Configured ``RandomForestClassifier`` instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs,
    )
