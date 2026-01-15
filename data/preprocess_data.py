import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(data_path: str):
    df = pd.read_csv(data_path)

    TARGET = "price_range"

    X = df.drop(TARGET, axis=1)
    y = df[TARGET].astype(int)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset columns: {list(df.columns)}")

    # In Mobile price dataset: all features are numeric
    numeric_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # Encoding, scaling
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
        ]
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Numeric cols:", len(numeric_cols))
    print("Train class balance:", y_train.value_counts(normalize=True).to_dict())
    print("Test  class balance:", y_test.value_counts(normalize=True).to_dict())

    return X_train, X_test, y_train, y_test, preprocess, numeric_cols, categorical_cols

def build_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

if __name__ == "__main__":
    data_path = "data/online_shoppers_intention.csv"
    preprocess_data(data_path)