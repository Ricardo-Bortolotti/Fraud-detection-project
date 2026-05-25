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
    """KNN para classificação de fraude."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
        **kwargs,
    )
