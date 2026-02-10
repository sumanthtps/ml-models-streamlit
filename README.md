# 📘 Assignment 2: Multi-Model Classification + Streamlit Deployment

This project is built for **BITS Pilani WILP (M.Tech AIML/DSE)** Assignment-2.
It implements and compares **6 classification models** on a single dataset, and deploys an interactive **Streamlit app** for inference and evaluation.

---

## 1) Problem Statement
Predict the **mobile price range** (`price_range`) using handset specifications from the Mobile Price Classification dataset.

### ✅ Assignment compliance checklist
- [x] One common dataset for all models (Mobile Price Classification)
- [x] 6 ML models implemented:
  1. Logistic Regression
  2. Decision Tree
  3. K-Nearest Neighbors (KNN)
  4. Naive Bayes (Gaussian)
  5. Random Forest (Ensemble)
  6. XGBoost (Ensemble)
- [x] Required metrics computed for each model:
  - Accuracy
  - AUC Score
  - Precision
  - Recall
  - F1 Score
  - MCC Score
- [x] Streamlit app with:
  - CSV upload
  - Model selection dropdown
  - Metric display
  - Confusion Matrix + Classification Report

---

## 2) Dataset Description
- **Dataset file used**: `data/mobile_price_train.csv`
- **Target variable**: `price_range`
- **Samples**: 2000+ (assignment minimum satisfied)
- **Features**: 20 numeric feature columns (assignment minimum feature count satisfied)

The preprocessing pipeline includes:
- Median imputation for numeric fields
- Standard scaling of all numerical features

---

## 3) Repository Structure

```text
ml-models-streamlit/
├── app.py
├── train.py
├── requirements.txt
├── metrics_comparison.csv
├── README.md
├── data/
│   ├── mobile_price_train.csv
│   ├── mobile_price_test.csv
│   └── preprocess_data.py
└── model/
    ├── logistic_regression.py
    ├── decision_tree.py
    ├── knn.py
    ├── naive_bayes.py
    ├── random_forest.py
    └── xgboost_model.py
```

---

## 4) Setup Instructions

### Create environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train all models and generate metrics
```bash
python train.py
```
This command will:
- Train all 6 models
- Save trained pipelines to `model/*.pkl`
- Generate `metrics_comparison.csv`

### Run Streamlit app
```bash
streamlit run app.py
```

---

## 5) Model Comparison Table (Current Run)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| KNN | 0.5175 | 0.7777 | 0.5267 | 0.5175 | 0.5199 | 0.3574 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest (Ensemble) | 0.8775 | 0.9809 | 0.8771 | 0.8775 | 0.8773 | 0.8367 |
| XGBoost (Ensemble) | 0.9350 | 0.9944 | 0.9347 | 0.9350 | 0.9348 | 0.9134 |

> Note: Exact values may differ slightly based on package version and hardware.

---

## 6) Observations About Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer in this run with top Accuracy/F1/MCC and near-perfect AUC after scaling. |
| Decision Tree | Easy to interpret but weaker AUC/MCC compared to ensemble methods, indicating limited class boundary flexibility. |
| KNN | Good raw accuracy but lower recall/F1, suggesting sensitivity to neighborhood structure and class overlap. |
| Naive Bayes | Very high recall but poor precision and MCC, meaning aggressive predictions lead to many false positives. |
| Random Forest (Ensemble) | Strong ensemble baseline with high AUC and stable all-round metrics, but below Logistic Regression/XGBoost in this run. |
| XGBoost (Ensemble) | Near-top performer with excellent AUC and MCC, showing robust ranking ability across all classes. |

---

## 7) Streamlit Features Implemented
- 📂 Upload test CSV
- 🤖 Select any one of the 6 trained models
- 📈 View model comparison table and metric-wise bar chart
- 🔮 Generate predictions and download output CSV
- 🧾 If target column is provided, view:
  - Accuracy, AUC, Precision, Recall, F1, MCC
  - Confusion matrix
  - Classification report
- 🧠 Auto-generated model-wise observations

---

## 8) Deployment (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud
3. Click **New App** and select this repo.
4. Set main file path as `app.py`.
5. Deploy.

---

## 9) Reproducibility Notes
- Random seed is fixed (`42`) for consistent model behavior.
- All models share the same train-test split and preprocessing pipeline.

---

## 10) Author
Prepared by **Sumanth_T_P** submission.
BITSId **20250AA5544**.