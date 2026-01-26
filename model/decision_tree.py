from sklearn.tree import DecisionTreeClassifier


def build_decision_tree(random_seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        class_weight="balanced",
        random_state=random_seed,
    )
