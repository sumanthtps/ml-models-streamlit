from sklearn.neighbors import KNeighborsClassifier


def build_knn(n_jobs: int = 1) -> KNeighborsClassifier:
    """Build a KNN classifier with explicit worker control."""
    return KNeighborsClassifier(
        n_neighbors=7,
        n_jobs=n_jobs,
    )