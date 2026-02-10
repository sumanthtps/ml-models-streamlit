from xgboost import XGBClassifier


def build_xgboost(
    random_seed: int,
    n_classes: int,
    n_jobs: int = 1,
    use_cuda: bool = False,
) -> XGBClassifier:
    """Build an XGBoost classifier for binary or multi-class data with optional CUDA."""
    common_kwargs = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_seed,
        n_jobs=n_jobs,
    )

    if use_cuda:
        common_kwargs |= {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
        }

    if n_classes == 2:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common_kwargs,
        )

    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        **common_kwargs,
    )