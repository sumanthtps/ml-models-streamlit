import pandas as pd

DATA_PATH = "data/data.csv"   # change filename
df = pd.read_csv(DATA_PATH)

print("Rows, Columns:", df.shape)
print("\nFirst 3 rows:\n", df.head(3))
print("\nColumn types:\n", df.dtypes)

# 1) Choose target column AFTER seeing df.columns
print("\nAll columns:\n", list(df.columns))

# 2) After you decide TARGET, set it here:
TARGET = None  # replace with "your_target_column"

if TARGET:
    print("\nTarget distribution:\n", df[TARGET].value_counts(dropna=False).head(20))
    feature_cols = [c for c in df.columns if c != TARGET]
    print("\nFeature count (excluding target):", len(feature_cols))
    print("\nMissing values (top 10):\n", df.isna().sum().sort_values(ascending=False).head(10))
else:
    print("\nSet TARGET after you identify the label column.")
