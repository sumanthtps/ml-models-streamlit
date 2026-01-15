import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/online_shoppers_intention.csv")

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
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ]
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Numeric cols:", len(numeric_cols), "Categorical cols:", len(categorical_cols))
print("Train class balance:", y_train.value_counts(normalize=True).to_dict())
print("Test  class balance:", y_test.value_counts(normalize=True).to_dict())

Xt = preprocess.fit_transform(X_train)
print("Transformed train shape:", Xt.shape)