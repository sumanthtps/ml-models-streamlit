from sklearn.neighbors import KNeighborsClassifier

def build_knn(n_jobs: int = 1) -> KNeighborsClassifier:
    """Create a KNN estimator with controlled worker usage."""
    return KNeighborsClassifier(
        n_neighbors=4,
        n_jobs=n_jobs,
    )