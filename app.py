import os
import pickle
import warnings
from typing import Any, Optional, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.exceptions import InconsistentVersionWarning # type: ignore
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


# Application configuration
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

CLASS_LABEL_DISPLAY: dict[str, str] = {
    "p": "poisonous(p)",
    "e": "edible(e)",
}

# Header
st.set_page_config(page_title="Mushroom Classification - Sumanth T P", page_icon="🍄", layout="wide")
st.title("🍄 Mushroom Classification")
st.caption(
    "This project evaluates six classifiers for mushroom safety prediction: "
    "Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, and XGBoost. "
    "Use this app to compare model metrics and generate predictions from test CSV files."
)

st.markdown(
    """
<style>
    .quick-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.8rem;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        background: rgba(120, 176, 255, 0.07);
    }
    .quick-card h4 {
        margin: 0;
        font-size: 1rem;
    }
    .quick-card p {
        margin: 0.35rem 0 0;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

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

def ensure_split_files() -> None:
    """Create train/test split files when they are missing."""
    if os.path.exists(DOWNLOAD_TRAIN_FILE) and os.path.exists(DOWNLOAD_TEST_FILE):
        return
    if os.path.exists(RAW_DATA_FILE):
        preprocess_data(
            RAW_DATA_FILE,
            train_out_path=DOWNLOAD_TRAIN_FILE,
            test_out_path=DOWNLOAD_TEST_FILE,
        )

@st.cache_data
def load_raw_data() -> Optional[pd.DataFrame]:
    if not os.path.exists(RAW_DATA_FILE):
        return None
    return pd.read_csv(RAW_DATA_FILE)

@st.cache_data
def load_metrics() -> Optional[pd.DataFrame]:
    if not os.path.exists(METRICS_PATH):
        return None
    return pd.read_csv(METRICS_PATH)


def _ensure_pickle_compat() -> None:
    """Backfill deprecated sklearn symbols required by legacy pickle artifacts."""
    try:
        from sklearn.compose import _column_transformer as ct_module  # type: ignore

        if not hasattr(ct_module, "_RemainderColsList"):

            class _RemainderColsList(list):
                pass

            ct_module._RemainderColsList = _RemainderColsList # type: ignore
    except Exception:
        pass


def _patch_simple_imputer_internals(estimator: Any) -> None:
    """Patch SimpleImputer internals for cross-version sklearn compatibility."""
    if isinstance(estimator, SimpleImputer):
        if not hasattr(estimator, "_fill_dtype"):
            try:
                estimator._fill_dtype = np.asarray(estimator.statistics_).dtype # type: ignore
            except Exception:
                estimator._fill_dtype = object # type: ignore

    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _patch_simple_imputer_internals(step)

    if isinstance(estimator, ColumnTransformer):
        if hasattr(estimator, "transformers_"):
            for _, transformer, _ in estimator.transformers_:
                _patch_simple_imputer_internals(transformer)


def _extract_expected_columns(model: Any) -> Optional[List[str]]:
    """Return the expected feature column order from the fitted estimator."""
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            return None

    if isinstance(model, Pipeline):
        for _, step in model.steps:
            cols = _extract_expected_columns(step)
            if cols:
                return cols

    if isinstance(model, ColumnTransformer) and hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_) # type: ignore
        except Exception:
            return None

    return None


def _align_dataframe_to_model(X: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Align inference features to the schema expected by the trained model."""
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
    """Resolve the preferred artifact path for a model key."""
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                try:
                    model = pickle.load(f)
                except AttributeError as exc:
                    if "_RemainderColsList" not in str(exc):
                        raise
                    _ensure_pickle_compat()
                    f.seek(0)
                    model = pickle.load(f)
        else:
            model = joblib.load(path)

    _patch_simple_imputer_internals(model)
    return model


def get_model_column(df: pd.DataFrame) -> str:
    return "ML Model Name" if "ML Model Name" in df.columns else ("ML Model name" if "ML Model name" in df.columns else "model")

def normalize_model_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def resolve_model_name_from_metrics(metrics_name: str) -> Optional[str]:
    if metrics_name in MODEL_KEYS:
        return metrics_name

    normalized_target = normalize_model_name(metrics_name)
    for ui_name in MODEL_KEYS:
        if normalize_model_name(ui_name) == normalized_target:
            return ui_name

    return None

def calculate_metrics(y_true: pd.Series, preds: np.ndarray, proba: Any, model: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "Accuracy": float(accuracy_score(y_true, preds)),
        "Precision": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, preds)),
    }

    # Compute binary ROC-AUC only when probabilities and class labels are available.
    if proba is not None and hasattr(model, "classes_") and len(getattr(model, "classes_", [])) == 2:
        classes = list(model.classes_)
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

def to_display_class_label(value: Any) -> Any:
    return CLASS_LABEL_DISPLAY.get(str(value), value)

ensure_split_files()
metrics_df = load_metrics()
dataset_context = load_dataset_context()
raw_df = load_raw_data()

if "selected_model_name" not in st.session_state:
    st.session_state["selected_model_name"] = list(MODEL_KEYS.keys())[0]
if "predict_data_source" not in st.session_state:
    st.session_state["predict_data_source"] = "Use default file"
if "uploader_reset_key" not in st.session_state:
    st.session_state["uploader_reset_key"] = 0
if "selected_model_name_pending" in st.session_state:
    st.session_state["selected_model_name"] = st.session_state.pop("selected_model_name_pending")

# Sidebar
with st.sidebar:
    st.header("Controls")
    model_options = list(MODEL_KEYS.keys())
    st.selectbox(
        "Select model",
        model_options,
        key="selected_model_name",
    )
    selected_model_name = st.session_state["selected_model_name"]
    st.markdown("---")
    st.caption("Workflow")
    st.markdown("1. Compare model metrics\n2. Predict using CSV\n3. Review observations")
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    if st.button("Use default test data", width="stretch"):
        st.session_state["predict_data_source"] = "Use default file"
        st.toast("Prediction source set to default test file.")

    if st.button("Reset uploaded CSV", width="stretch"):
        st.session_state["uploader_reset_key"] = st.session_state.get("uploader_reset_key", 0) + 1
        st.toast("Upload control reset.")

    if metrics_df is not None and "MCC" in metrics_df.columns:
        model_col = get_model_column(metrics_df)
        best_model_raw = str(metrics_df.sort_values("MCC", ascending=False).iloc[0][model_col])
        quick_pick_best = st.button("Pick best model (MCC)", width="stretch")
        if quick_pick_best:
            resolved_model_name = resolve_model_name_from_metrics(best_model_raw)
            if resolved_model_name is None:
                st.warning(
                    f"Could not match metrics model '{best_model_raw}' to selectable models. "
                    "Please select the model manually."
                )
            else:
                st.session_state["selected_model_name_pending"] = resolved_model_name
                st.toast(f"Selected model for this run: {resolved_model_name}")
                st.rerun()

# App content
with st.expander("Description", expanded=False):
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

    st.markdown("#### Study Framing")
    st.markdown(
        f"""
- **Sample / unit of analysis:** {dataset_context['sample_unit']}
- **Population:** {dataset_context['population']}
"""
    )

    if raw_df is not None and DEFAULT_TARGET in raw_df.columns:
        st.markdown("#### Data Quality & Balance Snapshot")

        total_missing = int(raw_df.isna().sum().sum())
        missing_pct = float((raw_df.isna().sum().sum() / raw_df.size) * 100) if raw_df.size else 0.0
        unique_features = int(raw_df.drop(columns=[DEFAULT_TARGET], errors="ignore").nunique().sum())

        q1, q2, q3 = st.columns(3)
        q1.metric("Missing values", f"{total_missing:,}")
        q2.metric("Missing (%)", f"{missing_pct:.2f}%")
        q3.metric("Total unique feature levels", f"{unique_features:,}")

        class_dist = raw_df[DEFAULT_TARGET].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
        class_dist["class"] = class_dist["class"].map(to_display_class_label)
        class_dist["percent"] = (class_dist["count"] / class_dist["count"].sum() * 100).round(2)

        b1, b2 = st.columns([1, 2])
        with b1:
            st.markdown("**Class Distribution**")
            st.dataframe(class_dist, width="stretch")
        with b2:
            fig_cls, ax_cls = plt.subplots(figsize=(6, 3.5))
            ax_cls.bar(class_dist["class"].astype(str), class_dist["count"])
            ax_cls.set_ylabel("Count")
            ax_cls.set_xlabel("Class")
            ax_cls.set_title("Target Distribution")
            st.pyplot(fig_cls)

        with st.expander("Feature sample (first 10 rows)", expanded=False):
            st.dataframe(raw_df.head(10), width="stretch")

overview_tab, predict_tab, insight_tab = st.tabs(["Model Comparison", "Predict", "Observations"])


with overview_tab:
    st.subheader("Model Metrics Table")
    if metrics_df is None:
        st.warning("metrics_comparison.csv not found. Run training first to generate it.")
    else:
        model_col = get_model_column(metrics_df)
        top_row = metrics_df.sort_values("MCC", ascending=False).iloc[0] if "MCC" in metrics_df.columns else metrics_df.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Top model", str(top_row[model_col]))
        if "Accuracy" in top_row:
            m2.metric("Top Accuracy", f"{float(top_row['Accuracy']):.3f}")
        if "AUC" in top_row and pd.notna(top_row["AUC"]):
            try:
                m3.metric("Top AUC", f"{float(top_row['AUC']):.3f}")
            except Exception:
                m3.metric("Top AUC", str(top_row["AUC"]))

        st.dataframe(metrics_df, width="stretch")
        metric_name = st.selectbox("Compare metric", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = metrics_df.sort_values(metric_name, ascending=False)

        model_col = get_model_column(plot_df)
        bar_colors = plt.cm.Set2(np.linspace(0, 1, len(plot_df))) # type: ignore
        ax.bar(plot_df[model_col].astype(str), plot_df[metric_name], color=bar_colors, width=0.65)
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
        key="predict_data_source",
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
        uploaded_file = st.file_uploader(
            "Upload CSV test data",
            type=["csv"],
            key=f"predict_upload_{st.session_state['uploader_reset_key']}",
        )
        if uploaded_file is None:
            st.info("Upload a CSV file to run predictions and generate evaluation charts.")
        else:
            df = pd.read_csv(uploaded_file)
            source_label = "uploaded file"

    if df is not None:
        st.write(f"Data preview ({source_label})")
        st.dataframe(df.head(10), width="stretch")

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
            output_df["ground_truth"] = pd.Series(y_true).map(to_display_class_label).values
        output_df["prediction"] = pd.Series(preds).map(to_display_class_label).values

        st.write("Predictions preview")
        st.dataframe(output_df.head(20), width="stretch")
        st.download_button(
            label="Download predictions CSV",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        # Metrics + charts
        if y_true is not None:
            st.markdown("---")
            st.subheader("Evaluation Metrics")

            metric_result = calculate_metrics(y_true, preds, proba, model)
            st.dataframe(pd.DataFrame([metric_result]), width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
                labels = list(getattr(model, "classes_", [])) or sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(preds)])))
                display_labels = [to_display_class_label(label) for label in labels]
                ConfusionMatrixDisplay.from_predictions(
                    y_true,
                    preds,
                    labels=labels,
                    display_labels=display_labels,
                    ax=ax_cm,
                    colorbar=False,
                )
                ax_cm.set_title(f"Confusion Matrix - {selected_model_name}")
                st.pyplot(fig_cm)

            with col2:
                st.markdown("**Classification Report**")
                target_names = [to_display_class_label(label) for label in labels]
                st.code(classification_report(y_true, preds, labels=labels, 
                                              target_names=target_names, zero_division=0), language="text")

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

        st.dataframe(pd.DataFrame(observations), width="stretch")
        
st.markdown("---")
st.caption("© 2026 Sumanth T P. All rights reserved.")
