import pandas as pd

from benchmark import benchmark

def main():
    df = pd.read_csv("Data/clean_train.csv")

    # === Get Benchmark Results ===
    acc, auc, class_report = benchmark(df)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(class_report)
    
if __name__ == "__main__":
    main()