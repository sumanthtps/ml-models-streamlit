# Assignment 2: Multi-Model Classification + Streamlit Deployment
It implements six classification models on one common dataset and provides a Streamlit app for interactive predictions and evaluation.

---

## 1) Problem Statement
Build a complete ML workflow for a classification problem by:
- training and comparing **6 required classifiers** on the **same dataset**,
- evaluating each model using the required metrics,
- building an interactive **Streamlit app** for predictions and evaluation,
- and preparing repository artifacts for deployment/submission.

In this project, the classification task is:
> **Predict mushroom class (`class`) from mushroom features** (edible vs poisonous categories in the selected dataset encoding).

---

## 2) Dataset Description
- **Dataset used:** `data/mushroom.csv`
- **Type:** Classification dataset
- **Target column:** `class`
- **Total columns:** 21 (20 feature columns + 1 target)
- **Rows available:** 61,069

### Train/Test preparation
- The project uses `data/preprocess_data.py` for preprocessing and splitting.
- Generated split files used by the app:
  - `data/mushroom_train.csv`
  - `data/mushroom_test.csv`

---

## 3) Models Used
The following six models are implemented (as required in the assignment):
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Model builder scripts are available under `model/`.

---

## 4) Evaluation Metrics (Required)
For every model, the following metrics are calculated:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5) Comparison Table (All 6 Models)
Latest run (from `metrics_comparison.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8444 | 0.9160 | 0.8427 | 0.8463 | 0.8435 | 0.6891 |
| Decision Tree | 0.9989 | 0.9989 | 0.9988 | 0.9989 | 0.9988 | 0.9977 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.6091 | 0.8367 | 0.7572 | 0.6471 | 0.5772 | 0.3891 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 0.9571 | 0.9945 | 0.9555 | 0.9587 | 0.9567 | 0.9142 |

---

## 6) Observations About Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline performance, but clearly below tree/ensemble models on this dataset. |
| Decision Tree | Very high scores across all metrics, indicating strong class separation with tree rules. |
| KNN | Perfect scores on this run; likely benefits from local neighborhood structure and preprocessing. |
| Naive Bayes | Lowest-performing model overall; assumptions appear less suitable for this dataset distribution. |
| Random Forest (Ensemble) | Perfect or near-perfect metrics, showing robust ensemble performance. |
| XGBoost (Ensemble) | Excellent all-round performance and high AUC/MCC, slightly below the perfect-score models in this run. |

---

## 7) Streamlit App Features Implemented
The app (`app.py`) includes all requested core features:
- **CSV upload option** for inference data
- **Model selection dropdown** (6 models)
- **Display of evaluation metrics**
- **Confusion matrix** and **classification report** (when target column is provided)
- Prediction preview and downloadable output CSV
- Model comparison table and metric-wise chart

Run locally with:

```bash
streamlit run app.py
```

---

## 8) Repository Structure

```text
ml-models-streamlit/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── metrics_comparison.csv
├── data/
│   ├── mushroom.csv
│   ├── mushroom_train.csv
│   ├── mushroom_test.csv
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

## 9) Setup and Execution

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train all models and generate metrics
```bash
python train.py
```
This will:
- train all 6 models,
- save model artifacts in `model/*.pkl`,
- create/update `metrics_comparison.csv`.

### Run Streamlit app
```bash
streamlit run app.py
```

---

## 10) Deployment Steps (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Open https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App**.
4. Select the repository and branch.
5. Set main file path as `app.py`.
6. Click **Deploy**.

---

## 11) Submission Notes (as per assignment template)
Include these in your final submitted PDF in order:
1. GitHub repository link
2. Live Streamlit app link
3. Screenshot of execution on BITS Virtual Lab
4. This README content (problem statement, dataset description, model comparison, and observations)

> Add your final links below before submission:
- **GitHub Repo Link:** _<add-your-link>_
- **Live Streamlit App Link:** _<add-your-link>_

---

## 12) Author
- **Name:** Sumanth_T_P
- **BITS ID:** 20250AA5544
