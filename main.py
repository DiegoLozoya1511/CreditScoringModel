import pandas as pd
from xgboost import XGBClassifier, plot_importance

from benchmark import benchmark
from benchmark import xgboost

def main():
    df = pd.read_csv("Data/clean_train.csv")

    # === Get Benchmark Results ===
    acc, auc, class_report = benchmark(df)
    acc_xgb, auc_xgb, class_report_xgb, xgb_model = xgboost(df)


    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(class_report)

    print(f"\nXGBoost Accuracy: {acc_xgb:.4f}")
    print(f"XGBoost AUC Score: {auc_xgb:.4f}")
    print("\nXGBoost Classification Report:")
    print(class_report_xgb)

    plot_importance(xgb_model)
    
if __name__ == "__main__":
    main()