import pandas as pd
import joblib

from utils import get_benchmark, data_splitter
from credit_model import get_weights, get_scores, scale_scores
from visualizations import plot_score_distribution


def main():
    # Load the cleaned training data
    df = pd.read_csv("Data/clean_train.csv")

    # === Get Benchmark ===
    model = joblib.load('Models/benchmark_model.pkl')
    acc, auc, class_report = get_benchmark(model, df)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(class_report)

    # === Feature Selection ===
    X_train, X_test, y_train, y_test = data_splitter(df, target='Credit_Score')
    
    # === WoE Encoded Data ===
    features = X_train.columns.tolist()
    X_train_woe = pd.read_csv('Data/X_train_woe.csv')
    X_test_woe = pd.read_csv('Data/X_test_woe.csv')
    
    # === Credit Model ===
    
    # --- Get feature weights from the model fitted on WoE-encoded data ---
    model = joblib.load('Models/weights_model.pkl')
    weights = get_weights(model, features)
    print(f"\nFeature Weights (sorted by importance):\n{weights}")

    # --- Calculate Scores ---
    # - Train -
    scores = get_scores(weights, X_train_woe)
    min_value = scores.min()
    train_scores = pd.DataFrame({
        'Credit_Score': y_train,
        'Score': scale_scores(scores, min_value)
    })
    plot_score_distribution(train_scores, 'Train')
    
    # - Test -
    scores = get_scores(weights, X_test_woe)
    test_scores = pd.DataFrame({
        'Credit_Score': y_test,
        'Score': scale_scores(scores, min_value)
    })
    plot_score_distribution(test_scores, 'Test')

if __name__ == "__main__":
    main()
