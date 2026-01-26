from sklearn.neighbors import KNeighborsClassifier


def build_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=7
    )
