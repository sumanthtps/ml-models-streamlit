from sklearn.ensemble import RandomForestClassifier


def build_random_forest(random_seed: int, n_jobs: int = 1) -> RandomForestClassifier:
    """Create a class-balanced random forest estimator."""
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=n_jobs,
    )