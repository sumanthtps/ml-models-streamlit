from sklearn.ensemble import RandomForestClassifier


def build_random_forest(random_seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
