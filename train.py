import os
from dataclasses import dataclass
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data.preprocess_data import preprocess_data

DATA_PATH = "data/online_shoppers_intention.csv"
TARGET = "Revenue"
RANDOM_SEED = 42
MODEL_DIR = "model"
METRICS_OUT_FILE = "metrics_comparison.csv"

@dataclass(frozen=True)
class Model:
    model_key: str
    display_name: str
    estimator: Any

def compute_scale_pos_weight(labels_train):
    negative_count = (labels_train == 0).sum()
    positive_count = (labels_train == 1).sum()
    return float(negative_count / positive_count)

def build_model_specs(scale_pos_weight):
    return [
        Model(
            model_key="logistic_regression",
            display_name="Logistic Regression",
            estimator=LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
        Model(
            model_key="decision_tree",
            display_name="Decision Tree",
            estimator=DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
        Model(
            model_key="knn",
            display_name="KNN",
            estimator=KNeighborsClassifier(
                n_neighbors=7,
            ),
        ),
        Model(
            model_key="naive_bayes",
            display_name="Naive Bayes",
            estimator=GaussianNB(),
        ),
        Model(
            model_key="random_forest",
            display_name="Random Forest",
            estimator=RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        Model(
            model_key="xgboost",
            display_name="XGBoost",
            estimator=XGBClassifier(
                n_estimators=700,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
            ),
        ),
    ]

def compute_binary_metrics(
        trained_pipeline: Pipeline,
        features_test: pd.DataFrame,
        labels_test: pd.Series,
):
    predicted_labels = trained_pipeline.predict(features_test)

    accuracy = float(accuracy_score(labels_test, predicted_labels))
    precision = float(precision_score(labels_test, predicted_labels, zero_division=0))
    recall = float(recall_score(labels_test, predicted_labels, zero_division=0))
    f1 = float(2 * precision * recall) / (precision + recall)
    mcc = float(matthews_corrcoef(labels_test, predicted_labels))

    predicted_probabilities = trained_pipeline.predict_proba(features_test)[:, 1]
    auc = float(roc_auc_score(labels_test, predicted_probabilities))

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "AUC": auc,

    }

def main():
    (features_train,
     features_test,
     labels_train,
     labels_test,
     preprocess_transformer,
     _numeric_cols,
     _categorical_cols
     ) = preprocess_data(DATA_PATH)

    scale_pos_weight = compute_scale_pos_weight(labels_train)
    model_specs = build_model_specs(scale_pos_weight)

    metric_rows= []

    for model_spec in model_specs:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocess_transformer),
                ("model", model_spec.estimator),
            ]
        )

        pipeline.fit(features_train, labels_train)

        metrics = compute_binary_metrics(
            pipeline,
            features_test,
            labels_test
        )

        metric_row: Dict[str, Any] = {"ML Model name": model_spec.display_name}
        metric_row.update(metrics)
        metric_rows.append(metric_row)

        model_file_path = os.path.join(MODEL_DIR, f"{model_spec.model_key}_pipeline.joblib")
        joblib.dump(pipeline, model_file_path)

        print(f"Trained and saved: {model_spec.display_name} to {model_file_path}")

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(METRICS_OUT_FILE, index=False)

    print("\nSaved metrics table", METRICS_OUT_FILE)
    print(metrics_table.sort_values("AUC", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()