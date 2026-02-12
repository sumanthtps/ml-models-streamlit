import os
import pickle
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data.preprocess_data import preprocess_data
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS_PATH = "metrics_comparison.csv"
MODEL_DIR = "model/saved_models/"
DEFAULT_TARGET = "class"
RAW_DATA_FILE = "data/mushroom.csv"
DEFAULT_PREDICT_FILE = "data/mushroom_test.csv"
DOWNLOAD_TRAIN_FILE = "data/mushroom_train.csv"
DOWNLOAD_TEST_FILE = "data/mushroom_test.csv"

MODEL_KEYS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}

st.set_page_config(page_title="Mushroom Classification", layout="wide")
st.title("🍄 Mushroom Classification - Edible or Poisonous")
st.caption("Built with 6 classifiers + model explorer")


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    if not os.path.exists(METRICS_PATH):
        return None
    return pd.read_csv(METRICS_PATH)


@st.cache_resource
def load_model(model_key: str) -> Any:
    for ext in (".pkl", "_pipeline.joblib", ".joblib"):
        path = os.path.join(MODEL_DIR, f"{model_key}{ext}" if ext.startswith(".") else f"{model_key}{ext}")
        if os.path.exists(path):
            if path.endswith(".pkl"):
                with open(path, "rb") as file:
                    return pickle.load(file)
            return __import__("joblib").load(path)
    raise FileNotFoundError(f"No model artifact found for key '{model_key}' under {MODEL_DIR}")




def ensure_split_files() -> None:
    if os.path.exists(DOWNLOAD_TRAIN_FILE) and os.path.exists(DOWNLOAD_TEST_FILE):
        return
    if os.path.exists(RAW_DATA_FILE):
        preprocess_data(RAW_DATA_FILE, train_out_path=DOWNLOAD_TRAIN_FILE, test_out_path=DOWNLOAD_TEST_FILE)


def get_model_column(df: pd.DataFrame) -> str:
    return "ML Model Name" if "ML Model Name" in df.columns else "ML Model name"


def calculate_metrics(y_true: pd.Series, preds: pd.Series, proba: Any | None) -> dict[str, float | str]:
    metrics = {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, average="macro", zero_division=0),
        "Recall": recall_score(y_true, preds, average="macro", zero_division=0),
        "F1": f1_score(y_true, preds, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, preds),
    }

    if proba is not None:
        try:
            if proba.shape[1] == 2:
                metrics["AUC"] = roc_auc_score(y_true, proba[:, 1])
            else:
                metrics["AUC"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        except ValueError:
            metrics["AUC"] = "N/A"
    else:
        metrics["AUC"] = "N/A"

    return metrics


with st.sidebar:
    st.header("⚙️ Controls")
    selected_model_name = st.selectbox("Select model", list(MODEL_KEYS.keys()))

ensure_split_files()
metrics_df = load_metrics()

overview_tab, predict_tab, insight_tab = st.tabs(["Model Comparison", "Predict", "Observations"])

with overview_tab:
    st.subheader("Model Metrics Table")
    if metrics_df is None:
        st.warning("metrics_comparison.csv not found. Run `python train.py` first.")
    else:
        st.dataframe(metrics_df, width="stretch")
        metric_name = st.selectbox("Compare metric", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_df = metrics_df.sort_values(metric_name, ascending=False)
        ax.bar(plot_df[get_model_column(plot_df)], plot_df[metric_name], color="#4f46e5")
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(f"{metric_name} by Model")
        st.pyplot(fig)

with predict_tab:
    st.subheader("Predict on Test CSV")
    st.caption("Use default split files or upload your own CSV, then run predictions and metrics.")

    dl_cols = st.columns(1)
    # if os.path.exists(DOWNLOAD_TRAIN_FILE):
    #     with open(DOWNLOAD_TRAIN_FILE, "rb") as file:
    #         dl_cols[0].download_button(
    #             label="Download 80% train split",
    #             data=file.read(),
    #             file_name=os.path.basename(DOWNLOAD_TRAIN_FILE),
    #             mime="text/csv",
    #         )
    if os.path.exists(DOWNLOAD_TEST_FILE):
        with open(DOWNLOAD_TEST_FILE, "rb") as file:
            dl_cols[1].download_button(
                label="Download 20% test split",
                data=file.read(),
                file_name=os.path.basename(DOWNLOAD_TEST_FILE),
                mime="text/csv",
            )

    data_source = st.radio(
        "Prediction data source",
        ("Use default file", "Upload CSV"),
        horizontal=True,
    )

    df = None
    source_label = ""
    if data_source == "Use default file":
        if os.path.exists(DEFAULT_PREDICT_FILE):
            df = pd.read_csv(DEFAULT_PREDICT_FILE)
            source_label = f"default file: {DEFAULT_PREDICT_FILE}"
        else:
            st.error(f"Default file not found: {DEFAULT_PREDICT_FILE}")
    else:
        uploaded_file = st.file_uploader("Upload CSV test data", type=["csv"], key="predict_upload")
        if uploaded_file is None:
            st.info("Upload a CSV file to run predictions and generate evaluation charts.")
        else:
            df = pd.read_csv(uploaded_file)
            source_label = "uploaded file"

    if df is not None:
        st.write(f"Data preview ({source_label})")
        st.dataframe(df.head(10), width="stretch")

        model = load_model(MODEL_KEYS[selected_model_name])

        y_true = None
        X_infer = df

        preds = model.predict(X_infer)
        proba = model.predict_proba(X_infer) if hasattr(model, "predict_proba") else None

        output_df = X_infer.copy()
        if y_true is not None:
            output_df["ground_truth"] = y_true.values
        output_df["prediction"] = preds

        st.write("Predictions preview")
        st.dataframe(output_df.head(20), width="stretch")
        st.download_button(
            label="Download predictions CSV",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Evaluation Metrics")
        if y_true is not None:
            metric_result = calculate_metrics(y_true, preds, proba)
            st.dataframe(pd.DataFrame([metric_result]), width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                ConfusionMatrixDisplay.from_predictions(y_true, preds, ax=ax_cm, colorbar=False)
                st.pyplot(fig_cm)

            with col2:
                st.markdown("**Classification Report**")
                st.code(classification_report(y_true, preds, zero_division=0), language="text")
        else:
            pass

with insight_tab:
    st.subheader("Model-wise Observations")
    if metrics_df is None:
        st.info("Train models to auto-generate performance observations.")
    else:
        model_col = get_model_column(metrics_df)
        best_auc = metrics_df.loc[metrics_df["AUC"].idxmax(), model_col]
        best_acc = metrics_df.loc[metrics_df["Accuracy"].idxmax(), model_col]
        best_mcc = metrics_df.loc[metrics_df["MCC"].idxmax(), model_col]

        st.success(f"Best AUC: **{best_auc}** | Best Accuracy: **{best_acc}** | Best MCC: **{best_mcc}**")

        observations = []
        for _, row in metrics_df.iterrows():
            model_name = row[model_col]
            line = (
                f"{model_name}: Accuracy={row['Accuracy']:.3f}, AUC={row['AUC']:.3f}, "
                f"F1={row['F1']:.3f}, MCC={row['MCC']:.3f}."
            )
            observations.append({"ML Model Name": model_name, "Observation about model performance": line})

        st.dataframe(pd.DataFrame(observations), width="stretch")