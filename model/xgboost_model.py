from xgboost import XGBClassifier


def build_xgboost(random_seed: int, n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_seed,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
