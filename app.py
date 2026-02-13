import os
import sys
import pickle
from typing import Any, Optional, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from data.preprocess_data import preprocess_data


# ----------------------------
# Config
# ----------------------------
METRICS_PATH: str = "metrics_comparison.csv"
MODEL_DIR: str = "model/saved_models"

DEFAULT_TARGET: str = "class"
RAW_DATA_FILE: str = "data/mushroom.csv"
DEFAULT_PREDICT_FILE: str = "data/mushroom_test.csv"
DOWNLOAD_TRAIN_FILE: str = "data/mushroom_train.csv"
DOWNLOAD_TEST_FILE: str = "data/mushroom_test.csv"

MODEL_KEYS: dict[str, str] = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}


# ----------------------------
# Streamlit UI setup
# ----------------------------
st.set_page_config(page_title="Mushroom Classification", layout="wide")
st.title("Mushroom Classification - Edible or Poisonous")

def load_dataset_context() -> dict[str, Any]:
    info: dict[str, Any] = {
        "raw_rows": "N/A",
        "raw_columns": "N/A",
        "target": DEFAULT_TARGET,
        "sample_unit": "One mushroom observation",
        "population": "All mushrooms represented in the source dataset",
        "train_rows": "N/A",
        "test_rows": "N/A",
    }

    if os.path.exists(RAW_DATA_FILE):
        raw_df = pd.read_csv(RAW_DATA_FILE)
        info["raw_rows"] = int(raw_df.shape[0])
        info["raw_columns"] = int(raw_df.shape[1])

    if os.path.exists(DOWNLOAD_TRAIN_FILE):
        train_df = pd.read_csv(DOWNLOAD_TRAIN_FILE)
        info["train_rows"] = int(train_df.shape[0])

    if os.path.exists(DOWNLOAD_TEST_FILE):
        test_df = pd.read_csv(DOWNLOAD_TEST_FILE)
        info["test_rows"] = int(test_df.shape[0])

    return info

# ----------------------------
# Helpers
# ----------------------------
def ensure_split_files() -> None:
    if os.path.exists(DOWNLOAD_TRAIN_FILE) and os.path.exists(DOWNLOAD_TEST_FILE):
        return
    if os.path.exists(RAW_DATA_FILE):
        preprocess_data(
            RAW_DATA_FILE,
            train_out_path=DOWNLOAD_TRAIN_FILE,
            test_out_path=DOWNLOAD_TEST_FILE,
        )


@st.cache_data
def load_metrics() -> Optional[pd.DataFrame]:
    if not os.path.exists(METRICS_PATH):
        return None
    return pd.read_csv(METRICS_PATH)


def _ensure_pickle_compat() -> None:
    """
    Backfill removed sklearn private classes used in old pickle artifacts.
    (This addresses common unpickle failures for older ColumnTransformer artifacts.)
    """
    try:
        from sklearn.compose import _column_transformer as ct_module  # type: ignore

        if not hasattr(ct_module, "_RemainderColsList"):

            class _RemainderColsList(list):
                pass

            ct_module._RemainderColsList = _RemainderColsList
    except Exception:
        pass


def _patch_simple_imputer_internals(estimator: Any) -> None:
    """
    Best-effort forward-compat patch for older pickled/joblib sklearn pipelines
    where SimpleImputer may miss some internal attributes in newer sklearn.

    Your stack trace ended around SimpleImputer.transform using self._fill_dtype.
    If it's missing, we set it based on statistics_ dtype, else fallback to object.
    """
    if isinstance(estimator, SimpleImputer):
        if not hasattr(estimator, "_fill_dtype"):
            try:
                estimator._fill_dtype = np.asarray(estimator.statistics_).dtype
            except Exception:
                estimator._fill_dtype = object

    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _patch_simple_imputer_internals(step)

    if isinstance(estimator, ColumnTransformer):
        # transformers_ exists after fit; if missing, nothing to patch.
        if hasattr(estimator, "transformers_"):
            for _, transformer, _ in estimator.transformers_:
                _patch_simple_imputer_internals(transformer)


def _extract_expected_columns(model: Any) -> Optional[List[str]]:
    """
    Extract expected input columns from a fitted sklearn pipeline/model.
    Most reliable is feature_names_in_ (available on many estimators since sklearn 1.0+).
    """
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            return None

    if isinstance(model, Pipeline):
        # Try each step
        for _, step in model.steps:
            cols = _extract_expected_columns(step)
            if cols:
                return cols

    if isinstance(model, ColumnTransformer) and hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            return None

    return None


def _align_dataframe_to_model(X: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Align inference dataframe to the columns the model was trained on:
    - add missing expected columns as NaN
    - drop unexpected extra columns
    - reorder columns to expected order

    This prevents ColumnTransformer failures when uploaded CSV schema differs.
    """
    expected_cols = _extract_expected_columns(model)
    if not expected_cols:
        return X

    X_aligned = X.copy()

    # Add missing columns
    missing = [c for c in expected_cols if c not in X_aligned.columns]
    for c in missing:
        X_aligned[c] = pd.NA

    # Drop extra columns
    extra = [c for c in X_aligned.columns if c not in expected_cols]
    if extra:
        X_aligned = X_aligned.drop(columns=extra)

    # Reorder
    X_aligned = X_aligned[expected_cols]

    return X_aligned


def _safe_find_model_path(model_key: str) -> str:
    """
    Prefer joblib artifacts first to avoid stale .pkl.
    """
    candidates = [
        os.path.join(MODEL_DIR, f"{model_key}_pipeline.joblib"),
        os.path.join(MODEL_DIR, f"{model_key}.joblib"),
        os.path.join(MODEL_DIR, f"{model_key}.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No model artifact found for key '{model_key}' under {MODEL_DIR}")


@st.cache_resource
def load_model(model_key: str) -> Any:
    path = _safe_find_model_path(model_key)

    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            try:
                model = pickle.load(f)
            except AttributeError as exc:
                # Backfill common sklearn private symbol if needed
                if "_RemainderColsList" not in str(exc):
                    raise
                _ensure_pickle_compat()
                f.seek(0)
                model = pickle.load(f)
    else:
        model = joblib.load(path)

    # Apply forward-compat patch for SimpleImputer internals
    _patch_simple_imputer_internals(model)
    return model


def get_model_column(df: pd.DataFrame) -> str:
    return "ML Model Name" if "ML Model Name" in df.columns else ("ML Model name" if "ML Model name" in df.columns else "model")


def calculate_metrics(y_true: pd.Series, preds: np.ndarray, proba: Any, model: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "Accuracy": float(accuracy_score(y_true, preds)),
        "Precision": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, preds)),
    }

    # AUC for binary classification (handle string labels safely)
    if proba is not None and hasattr(model, "classes_") and len(getattr(model, "classes_", [])) == 2:
        classes = list(model.classes_)
        # For mushroom datasets, 'p' (poisonous) is a common "positive" label.
        pos_label = "p" if "p" in classes else classes[1]
        pos_index = classes.index(pos_label)

        try:
            y_bin = (y_true == pos_label).astype(int)
            metrics["AUC"] = float(roc_auc_score(y_bin, proba[:, pos_index]))
        except Exception:
            metrics["AUC"] = "N/A"
    else:
        metrics["AUC"] = "N/A"

    return metrics


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Controls")
    selected_model_name = st.selectbox("Select model", list(MODEL_KEYS.keys()))
    st.markdown("---")
    st.caption("Runtime info")
    st.code(
        f"python: {sys.version.split()[0]}\n"
        f"platform: {sys.platform}\n",
        language="text",
    )


# ----------------------------
# App
# ----------------------------
ensure_split_files()
metrics_df = load_metrics()
dataset_context = load_dataset_context()

with st.expander("Description", expanded=True):
    st.markdown("#### Problem Statement")
    st.write("Predict whether a mushroom is edible or poisonous using supervised classification.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total dataset rows", dataset_context["raw_rows"])
    c2.metric("Train set rows", dataset_context["train_rows"])
    c3.metric("Test set rows", dataset_context["test_rows"])

    st.markdown("#### Dataset Details")
    st.markdown(
        f"""
- **Dataset path:** `{RAW_DATA_FILE}`
- **Target column:** `{dataset_context['target']}`
- **Total columns:** {dataset_context['raw_columns']}
"""
    )
    st.info("Due to limitation of streamlit, took sample of 12k records(stratified)" \
    " from original dataset which is arround 67k", icon="ℹ️")

overview_tab, predict_tab, insight_tab = st.tabs(["Model Comparison", "Predict", "Observations"])


with overview_tab:
    st.subheader("Model Metrics Table")
    if metrics_df is None:
        st.warning("metrics_comparison.csv not found. Run training first to generate it.")
    else:
        st.dataframe(metrics_df, use_container_width=True)
        metric_name = st.selectbox("Compare metric", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])

        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_df = metrics_df.sort_values(metric_name, ascending=False)

        model_col = get_model_column(plot_df)
        ax.bar(plot_df[model_col].astype(str), plot_df[metric_name])
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(f"{metric_name} by Model")
        st.pyplot(fig)


with predict_tab:
    st.subheader("Predict on Test CSV")
    st.caption("Use default split files or upload your own CSV, then run predictions and metrics.")

    if os.path.exists(DOWNLOAD_TEST_FILE):
        with open(DOWNLOAD_TEST_FILE, "rb") as f:
            st.download_button(
                label="Download test split",
                data=f.read(),
                file_name=os.path.basename(DOWNLOAD_TEST_FILE),
                mime="text/csv",
            )

    data_source = st.radio(
        "Prediction data source",
        ("Use default file", "Upload CSV"),
        horizontal=True,
    )

    df: Optional[pd.DataFrame] = None
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
        st.dataframe(df.head(10), use_container_width=True)

        # Load model safely
        try:
            model = load_model(MODEL_KEYS[selected_model_name])
        except Exception as exc:
            st.error("Failed to load model artifact.")
            st.exception(exc)
            st.stop()

        # Extract y_true if present, and build X for inference
        y_true = df[DEFAULT_TARGET] if DEFAULT_TARGET in df.columns else None
        X_infer = df.drop(columns=[DEFAULT_TARGET], errors="ignore")

        # Align schema to what the pipeline expects
        X_infer = _align_dataframe_to_model(X_infer, model)

        # Predict
        try:
            preds = model.predict(X_infer)
            proba = model.predict_proba(X_infer) if hasattr(model, "predict_proba") else None
        except Exception as exc:
            st.error(
                "Prediction failed inside preprocessing (ColumnTransformer / SimpleImputer). "
                "This is often caused by scikit-learn version mismatch between training and deployment "
                "or schema mismatch in the uploaded CSV."
            )
            st.exception(exc)
            st.stop()

        output_df = X_infer.copy()
        if y_true is not None:
            output_df["ground_truth"] = y_true.values
        output_df["prediction"] = preds

        st.write("Predictions preview")
        st.dataframe(output_df.head(20), use_container_width=True)
        st.download_button(
            label="Download predictions CSV",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        # Metrics + charts if ground truth exists
        if y_true is not None:
            st.markdown("---")
            st.subheader("Evaluation Metrics")

            metric_result = calculate_metrics(y_true, preds, proba, model)
            st.dataframe(pd.DataFrame([metric_result]), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                ConfusionMatrixDisplay.from_predictions(y_true, preds, ax=ax_cm, colorbar=False)
                st.pyplot(fig_cm)

            with col2:
                st.markdown("**Classification Report**")
                st.code(classification_report(y_true, preds, zero_division=0), language="text")


with insight_tab:
    st.subheader("Model-wise Observations")
    if metrics_df is None:
        st.info("Train models to auto-generate performance observations.")
    else:
        model_col = get_model_column(metrics_df)
        best_auc = metrics_df.loc[metrics_df["AUC"].idxmax(), model_col] if "AUC" in metrics_df.columns else "N/A"
        best_acc = metrics_df.loc[metrics_df["Accuracy"].idxmax(), model_col] if "Accuracy" in metrics_df.columns else "N/A"
        best_mcc = metrics_df.loc[metrics_df["MCC"].idxmax(), model_col] if "MCC" in metrics_df.columns else "N/A"

        st.success(f"Best AUC: {best_auc} | Best Accuracy: {best_acc} | Best MCC: {best_mcc}")

        observations = []
        for _, row in metrics_df.iterrows():
            model_name = row[model_col]
            line_parts = []
            for k in ["Accuracy", "AUC", "F1", "MCC"]:
                if k in row and pd.notna(row[k]):
                    try:
                        line_parts.append(f"{k}={float(row[k]):.3f}")
                    except Exception:
                        line_parts.append(f"{k}={row[k]}")
            observations.append(
                {"ML Model Name": model_name, "Observation about model performance": f"{model_name}: " + ", ".join(line_parts)}
            )

        st.dataframe(pd.DataFrame(observations), use_container_width=True)
