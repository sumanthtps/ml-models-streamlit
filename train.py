import os
import pickle
from dataclasses import dataclass
from typing import Any, List, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from data.preprocess_data import preprocess_data
from model.logistic_regression import build_logistic_regression
from model.decision_tree import build_decision_tree
from model.knn import build_knn
from model.naive_bayes import build_naive_bayes
from model.random_forest import build_random_forest
from model.xgboost_model import build_xgboost

DATA_PATH = "data/mobile_price_train.csv"
RANDOM_SEED = 42

MODEL_DIR = "model"
METRICS_OUT_FILE = "metrics_comparison.csv"

@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    display_name: str
    estimator: Any


def get_model_specs(random_seed: int, n_classes: int) -> List[ModelSpec]:
    return [
        ModelSpec("logistic_regression", "Logistic Regression", build_logistic_regression(random_seed)),
        ModelSpec("decision_tree", "Decision Tree", build_decision_tree(random_seed)),
        ModelSpec("knn", "KNN", build_knn()),
        ModelSpec("naive_bayes", "Naive Bayes", build_naive_bayes()),
        ModelSpec("random_forest", "Random Forest", build_random_forest(random_seed)),
        ModelSpec("xgboost", "XGBoost", build_xgboost(random_seed, n_classes)),
    ]

def compute_metrics(
    trained_pipeline: Pipeline,
    features_test: pd.DataFrame,
    labels_test: pd.Series,
):
    predicted_labels = trained_pipeline.predict(features_test)

    accuracy = float(accuracy_score(labels_test, predicted_labels))
    precision = float(precision_score(labels_test, predicted_labels, average="macro", zero_division=0))
    recall = float(recall_score(labels_test, predicted_labels, average="macro", zero_division=0))
    f1 = float(f1_score(labels_test, predicted_labels, average="macro", zero_division=0))
    proba = trained_pipeline.predict_proba(features_test)
    auc = float(roc_auc_score(labels_test, proba, multi_class="ovr", average="macro"))
    mcc = float(matthews_corrcoef(labels_test, predicted_labels))

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
    }

def save_pipeline_as_pkl(pipeline: Pipeline, file_path: str) -> None:
    with open(file_path, "wb") as file:
        pickle.dump(pipeline, file)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    (
        features_train,
        features_test,
        labels_train,
        labels_test,
        preprocess_transformer,
    ) = preprocess_data(DATA_PATH)

    n_classes = int(labels_train.nunique())

    model_specs = get_model_specs(RANDOM_SEED, n_classes)

    metric_rows = []

    for model_spec in model_specs:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocess_transformer),
                ("model", model_spec.estimator),
            ]
        )

        pipeline.fit(features_train, labels_train)

        metrics = compute_metrics(pipeline, features_test, labels_test)

        row: Dict[str, Any] = {"Model name": model_spec.display_name}
        row.update(metrics)
        metric_rows.append(row)

        model_file_path = os.path.join(MODEL_DIR, model_spec.model_key + ".pkl")
        save_pipeline_as_pkl(pipeline, model_file_path)

        print(f"Trained and saved: {model_spec.display_name} to {model_file_path}")

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(METRICS_OUT_FILE, index=False)

    print("\nSaved metrics table", METRICS_OUT_FILE)
    print(metrics_table.sort_values("AUC", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
