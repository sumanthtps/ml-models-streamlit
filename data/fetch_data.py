import pandas as pd
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=468)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

print("Shape (rows, cols):", df.shape)
print("\nColumns:\n", list(df.columns))

TARGET = "Revenue"
print("\nTarget distribution:\n", df[TARGET].value_counts(dropna=False))
print("\nMissing values total:", int(df.isna().sum().sum()))

print("\nDtypes summary:\n", df.dtypes.value_counts())
print("\nSample rows:\n", df.head(3))

df.to_csv("data/online_shoppers_intention.csv", index=False)
print("\nSaved to: data/online_shoppers_intention.csv")
