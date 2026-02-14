# Assignment 2 – Multi-Model Classification with Streamlit

---

## 1) Problem Statement
Build a Machine Learning model that predicts whether a mushroom is **poisonous or edible** based on its physical and environmental attributes. The goal is to help identify potentially harmful mushrooms early so safer decisions can be made while handling or consuming them.

This application evaluates multiple classification models to determine the outcome:

- `e` → **Edible Mushroom**
- `p` → **Poisonous Mushroom**

---

## 2) Dataset Description
This project uses a mushroom classification dataset containing real-world style observations of mushroom specimens. The dataset includes cap, gill, stem, veil, ring, and habitat-related features that are highly useful for predicting whether a mushroom is edible or poisonous.

### Dataset Source
The dataset used in this project is available locally in this repository at: `data/mushroom.csv`

### Dataset Overview
- **Total Records:** 12,214
- **Total Columns:** 21
- **Input Features:** 20
- **Target Column:** `class`

### Attribute Details
- `cap-diameter`: diameter of the mushroom cap
- `cap-shape`: shape of the cap
- `cap-surface`: texture of the cap surface
- `cap-color`: color of the cap
- `does-bruise-or-bleed`: whether the mushroom bruises or bleeds
- `gill-attachment`: type of gill attachment
- `gill-spacing`: spacing between gills
- `gill-color`: color of the gills
- `stem-height`: height of the stem
- `stem-width`: width of the stem
- `stem-root`: root characteristic of the stem
- `stem-surface`: texture of the stem surface
- `stem-color`: color of the stem
- `veil-type`: type of veil present
- `veil-color`: color of the veil
- `has-ring`: whether a ring is present
- `ring-type`: type of ring
- `spore-print-color`: color of the spore print
- `habitat`: natural habitat of the mushroom
- `season`: season in which the mushroom appears
- `class`: mushroom class (`e` = edible, `p` = poisonous)

### 2.3 Train/Test split used
Data is split using stratified train-test split (`test_size=0.2`, `random_state=42`):
- **Train set:** 9,771 rows
- **Test set:** 2,443 rows

Data files:
- `data/mushroom_train.csv`
- `data/mushroom_test.csv`

---

## 3) Models Used and Evaluation Metrics
### 3.1 Models implemented
- Logistic Regression (`model/logistic_regression.py`)
- Decision Tree (`model/decision_tree.py`)
- KNN (`model/knn.py`)
- Naive Bayes (`model/naive_bayes.py`)
- Random Forest (`model/random_forest.py`)
- XGBoost (`model/xgboost_model.py`)

### 3.2 Evaluation metrics
All 6 models are evaluated on:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

---

## 4) Comparison Table (All 6 Models)
The following values are populated from `metrics_comparison.csv`:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8420 | 0.9184 | 0.8403 | 0.8438 | 0.8411 | 0.6841 |
| Decision Tree | 0.9947 | 0.9948 | 0.9945 | 0.9948 | 0.9946 | 0.9892 |
| KNN | 0.9992 | 1.0000 | 0.9992 | 0.9992 | 0.9992 | 0.9983 |
| Naive Bayes | 0.6013 | 0.8357 | 0.7550 | 0.6402 | 0.5666 | 0.3782 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 0.9476 | 0.9913 | 0.9462 | 0.9482 | 0.9471 | 0.8944 |

---

## 5) Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good baseline performance and balanced precision/recall; lower than tree/ensemble methods on this dataset. |
| Decision Tree | Very strong performance across all metrics; captures non-linear splits effectively. |
| KNN | Near-perfect scores, indicating strong local separability of classes after preprocessing. |
| Naive Bayes | Lowest overall scores; conditional independence assumption appears less suitable for this data distribution. |
| Random Forest (Ensemble) | Best overall performance (perfect metrics in current run), indicating excellent robustness and generalization on this split. |
| XGBoost (Ensemble) | Excellent AUC and MCC; slightly below Random Forest/KNN but still high-performing and reliable. |

---

## 6) Streamlit App Features
The Streamlit app (`app.py`) includes all required components:
- ✅ Dataset upload option (CSV)
- ✅ Model selection dropdown (multiple models)
- ✅ Display of evaluation metrics
- ✅ Confusion matrix / classification report

Additional implemented capabilities:
- Prediction preview table
- Downloadable prediction CSV
- Model comparison table and metric visualization

Run locally:

```bash
streamlit run app.py
```

---

## 7) Repository Structure (Required)

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── README_ASSIGNMENT.md
├── train.py
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
    ├── xgboost_model.py
    └── saved_models/
        ├── logistic_regression.pkl
        ├── decision_tree.pkl
        ├── knn.pkl
        ├── naive_bayes.pkl
        ├── random_forest.pkl
        └── xgboost.pkl
```

---

## 8) Requirements and Execution
Install dependencies:

```bash
pip install -r requirements.txt
```

Train all models and regenerate metrics:

```bash
python train.py
```

Run Streamlit app:

```bash
streamlit run app.py
```

---

## 9) Student Details
- **Name:** Sumanth_T_P
- **BITS ID:** 20250AA5544
- **Streamlit App Link:** `https://ml-models-app-sumanth-tp-2025aa05544.streamlit.app/`