from sklearn.linear_model import LogisticRegression

def build_logistic_regression(random_seed: int) -> LogisticRegression:
    """Create a class-balanced logistic regression estimator."""
    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=random_seed,
    )
