from sklearn.tree import DecisionTreeClassifier

def build_decision_tree(random_seed: int) -> DecisionTreeClassifier:
    """Create a class-balanced decision tree estimator."""
    return DecisionTreeClassifier(
        class_weight="balanced",
        random_state=random_seed,
    )
