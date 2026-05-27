"""KNN estimator factory."""

from typing import Any

from sklearn.neighbors import KNeighborsClassifier


def build_knn(
    n_neighbors: int = 5,
    weights: str = "uniform",
    algorithm: str = "auto",
    metric: str = "minkowski",
    p: int = 2,
    n_jobs: int = -1,
    **kwargs: Any,
) -> KNeighborsClassifier:
    """Build a KNN classifier for fraud detection.

    Args:
        n_neighbors: Number of neighbors.
        weights: Weight function (``"uniform"`` or ``"distance"``).
        algorithm: Neighbor search algorithm.
        metric: Distance metric.
        p: Power parameter for Minkowski metric.
        n_jobs: Parallel jobs; ``-1`` uses all cores.
        **kwargs: Additional arguments passed to ``KNeighborsClassifier``.

    Returns:
        Configured ``KNeighborsClassifier`` instance.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
        **kwargs,
    )
