import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from typing import Any, Dict


class XGBWithLabelEncoding(BaseEstimator, ClassifierMixin):
    """XGBoost classifier wrapper that transparently encodes string labels."""

    def __init__(self, **xgb_kwargs: Any) -> None:
        self.xgb_kwargs = xgb_kwargs

    def fit(self, X: Any, y: Any) -> "XGBWithLabelEncoding":
        self._label_encoder = LabelEncoder()
        encoded_y = self._label_encoder.fit_transform(y)
        self._model = XGBClassifier(**self.xgb_kwargs)
        self._model.fit(X, encoded_y)
        self.classes_ = self._label_encoder.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded_predictions = self._model.predict(X)
        return self._label_encoder.inverse_transform(encoded_predictions)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"xgb_kwargs": self.xgb_kwargs}

    def set_params(self, **params: Any) -> "XGBWithLabelEncoding":
        if "xgb_kwargs" in params:
            self.xgb_kwargs = params["xgb_kwargs"]
        return self


def build_xgboost(
    random_seed: int,
    n_classes: int,
    n_jobs: int = 1,
    use_cuda: bool = False,
) -> XGBWithLabelEncoding:
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
        return XGBWithLabelEncoding(
            objective="binary:logistic",
            eval_metric="logloss",
            **common_kwargs,
        )

    return XGBWithLabelEncoding(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        **common_kwargs,
    )
