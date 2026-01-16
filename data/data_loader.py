import pandas as pd
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo


def fetch_dataset() -> tuple[DataFrame, DataFrame, DataFrame]:
    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    # Dataset exploration
    print("="*60)
    print("Dataset: ")
    print("Online shoppers intention dataset")
    print("=" * 60)
    print("Shape (rows, cols):", df.shape)
    print("Columns:\n", list(df.columns))

    TARGET = "Revenue"
    print("\n" + "-" * 60)
    print("TARGET DISTRIBUTION:")
    print("-" * 60)
    print("\nTarget distribution:\n", df[TARGET].value_counts(dropna=False))
    print("\nMissing values total:", int(df.isna().sum().sum()))

    print("\nDtypes summary:\n", df.dtypes.value_counts())
    print("\nSample rows:\n", df.head(3))

    df.to_csv("data/online_shoppers_intention.csv", index=False)
    print("\nSaved to: data/online_shoppers_intention.csv")
    return df, X, y

if __name__ == "__main__":
    df, X, y = fetch_dataset()
