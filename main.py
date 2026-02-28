import pandas as pd

from benchmark import benchmark
from utils import data_splitter


def main():
    # Load the cleaned training data
    df = pd.read_csv("Data/clean_train.csv")

    # === Get Benchmark Results ===
    acc, auc, class_report = benchmark(df)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(class_report)

    # === Feature Selection ===
    X_train, X_test, y_train, y_test = data_splitter(df, target='Credit_Score')
    print(X_train.columns, len(X_train.columns))


if __name__ == "__main__":
    main()
