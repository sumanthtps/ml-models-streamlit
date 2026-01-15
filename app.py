import os
import joblib
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

METRICS_PATH: str = "metrics_comparison.csv"
MODEL_DIR: str = "model"

MODEL_KEYS = {
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    "Decision Tree": "decision_tree_pipeline.joblib",
    "KNN": "knn_pipeline.joblib",
    "Naive Bayes": "naive_bayes_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
    "XGBoost": "xgboost_pipeline.joblib",
}

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 - Online Shoppers Intention")

@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    if not os.path.exists(METRICS_PATH):
        return None
    return pd.read_csv(METRICS_PATH)

@st.cache_resource
def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)

metrics_df = load_metrics()

with st.sidebar:
    st.header("Controls")
    selected_model_name = st.selectbox("Select model", list(MODEL_KEYS.keys()))
    uploaded_file = st.file_uploader("Upload CSV (test data)", type=["csv"])
    target_col = st.text_input("Target column (optional)", value="Revenue")

st.subheader("Training Metrics")
if metrics_df is None:
    st.warning("metrics_comparison.csv not found. Run train.py first.")
else:
    # Works even if your column is currently 'ML Model name'
    model_col = "ML Model Name" if "ML Model Name" in metrics_df.columns else "ML Model name"
    filtered = metrics_df[metrics_df[model_col] == selected_model_name]
    st.dataframe(filtered if len(filtered) else metrics_df, use_container_width=True)

st.markdown("---")
st.subheader("Predict on Uploaded CSV")

if uploaded_file is None:
    st.info("Upload a CSV to run predictions.")
else:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    model = load_model(MODEL_KEYS[selected_model_name])

    y_true = None
    if target_col in df.columns:
        y_true = df[target_col].astype(int)
        X_infer = df.drop(columns=[target_col])
    else:
        X_infer = df

    preds = model.predict(X_infer)

    output_df = df.copy()
    output_df["prediction"] = preds

    st.write("Predictions preview:")
    st.dataframe(output_df.head(20), use_container_width=True)

    st.download_button(
        label="Download predictions CSV",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

    if y_true is not None:
        st.markdown("---")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ConfusionMatrixDisplay.from_predictions(y_true, preds, ax=ax, colorbar=False)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_true, preds, zero_division=0))
