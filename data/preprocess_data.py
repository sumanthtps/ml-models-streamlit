import pandas as pd
<<<<<<< HEAD
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(data_path: str):
    df = pd.read_csv(data_path)

    TARGET = "Revenue"

    X = df.drop(TARGET, axis=1)
    y = df[TARGET].astype(int)

    categorical_cols =  [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    ]

    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X[categorical_cols] = X[categorical_cols].astype(str)

    # The dataset contains imbalanced data
    # False- 10422
    # True = 1908
    # So we need to stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # Encoding, scaling
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", build_onehot()),
    ])

    preprocess = ColumnTransformer(
=======
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "class"
TRAIN_SPLIT_PATH = "data/mushroom_train.csv"
TEST_SPLIT_PATH = "data/mushroom_test.csv"


def preprocess_data(
    data_path: str,
    train_out_path: str = TRAIN_SPLIT_PATH,
    test_out_path: str = TEST_SPLIT_PATH,
):
    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols = list(X.select_dtypes(include=["number"]).columns)
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[TARGET] = y_train
    train_df.to_csv(train_out_path, index=False)

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(test_out_path, index=False)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess_transformer = ColumnTransformer(
>>>>>>> 39c0d60ecbed6ae5fb5433a360a49e4e5f0a2654
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Numeric cols:", len(numeric_cols), "Categorical cols:", len(categorical_cols))
    print("Train class balance:", y_train.value_counts(normalize=True).to_dict())
    print("Test  class balance:", y_test.value_counts(normalize=True).to_dict())
<<<<<<< HEAD

    return X_train, X_test, y_train, y_test, preprocess, numeric_cols, categorical_cols

def build_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

if __name__ == "__main__":
    data_path = "data/online_shoppers_intention.csv"
    preprocess_data(data_path)
=======
    print(f"Saved train split to {train_out_path}")
    print(f"Saved test split to {test_out_path}")

    return X_train, X_test, y_train, y_test, preprocess_transformer


if __name__ == "__main__":
    preprocess_data("data/mushroom.csv")
>>>>>>> 39c0d60ecbed6ae5fb5433a360a49e4e5f0a2654
