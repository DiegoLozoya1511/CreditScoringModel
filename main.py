import pandas as pd
import joblib

from utils import get_benchmark, data_splitter
from credit_model import get_weights, scores_df, treshold_classification
from visualizations import plot_score_distribution, plot_score_comparison, plot_confusion_matrix


def main():
    # Load the cleaned training data
    df = pd.read_csv("Data/clean_train.csv")

    # === Get Benchmark ===
    print("\n=== Benchmark Evaluation ===")
    model = joblib.load('Models/benchmark_model.pkl')
    acc, class_report = get_benchmark(model, df)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{class_report}")

    # === Feature Selection ===
    X_train, X_test, y_train, y_test = data_splitter(df, target='Credit_Score')

    # === WoE Encoded Data ===
    features = X_train.columns.tolist()

    # === Credit Model ===
    print("\n=== Credit Model Evaluation ===")

    # --- Get feature weights from the model fitted ---
    model = joblib.load('Models/weights_model.pkl')
    weights = get_weights(model, features)
    print(f"\nFeature Weights (sorted by importance):\n{weights}\n")

    # --- Calculate Scores ---
    train_scores, test_scores = scores_df(
        weights, X_train, y_train, X_test, y_test)

    # --- Treshold Classification ---
    train_scores, test_scores, tresholds = treshold_classification(
        train_scores, test_scores)

    # === Visualizations ===
    plot_score_distribution(train_scores, tresholds, 'Train')
    plot_score_distribution(test_scores, tresholds, 'Test')

    plot_score_comparison(train_scores, 'Train')
    plot_score_comparison(test_scores, 'Test')

    train_class_report = plot_confusion_matrix(train_scores, 'Train')
    test_class_report = plot_confusion_matrix(test_scores, 'Test')

    print(f"\nTrain Classification Report:\n{train_class_report}")
    print(f"\nTest Classification Report:\n{test_class_report}")


if __name__ == "__main__":
    main()
