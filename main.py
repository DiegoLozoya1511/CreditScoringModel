import pandas as pd
import joblib

from utils import get_benchmark, data_splitter
from credit_model import get_weights, get_scores, scale_scores
from visualizations import plot_score_distribution

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report



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
   
    
    # === Credit Model ===
    
    # --- Get feature weights from the model fitted on WoE-encoded data ---
    model = joblib.load('Models/weights_model.pkl')
    weights = get_weights(model, features)
    print(f"\nFeature Weights (sorted by importance):\n{weights}")

    # --- Calculate Scores ---
    # - Train -
    scores = get_scores(weights, X_train) # X_train_woe
    min_value = scores.min()
    train_scores = pd.DataFrame({
        'Credit_Score': y_train,
        'Score': scale_scores(scores, min_value)
    }).reset_index(drop=True)
    plot_score_distribution(train_scores, 'Train')
    
    # - Test -
    scores = get_scores(weights, X_test) # X_test_woe
    test_scores = pd.DataFrame({
        'Credit_Score': y_test,
        'Score': scale_scores(scores, min_value)
    }).reset_index(drop=True)
    plot_score_distribution(test_scores, 'Test')
    
    # --- Treshold Analysis ---
    t1 = 220
    
    t2 = 410
        
    train_scores['class'] = train_scores['Score'].apply(lambda x: 0 if x <= t1 else 1 if x < t2 else 2)
    test_scores['class']= test_scores['Score'].apply(lambda x: 0 if x <= t1 else 1 if x < t2 else 2)
    
    train_acc = accuracy_score(train_scores['Credit_Score'], train_scores['class'])
    test_acc = accuracy_score(test_scores['Credit_Score'], test_scores['class'])
    
    print(f"\nTrain Accuracy at thresholds {t1}, {t2}: {train_acc:.4f}")
    print(f"Test Accuracy at thresholds {t1}, {t2}: {test_acc:.4f}")

if __name__ == "__main__":
    main()
