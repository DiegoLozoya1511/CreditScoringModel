import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def benchmark(df: pd.DataFrame) -> tuple[float, float, str]:
    """
    Benchmark the performance of a logistic regression model on the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data for training and testing.
        
    Returns:
        tuple[float, float, str]: A tuple containing the accuracy, AUC score, and classification report.
    """
    df = df.copy()

    df['Credit_Score'] = df['Credit_Score'].map({
        'Poor': 0,
        'Standard': 1,
        'Good': 2
    })

    target = 'Credit_Score'
    y = df[target]

    X = df.drop(columns=["Customer_ID", "ID", "Name", "SSN", "Month", target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    class_report = classification_report(y_test, y_pred)

    return acc, auc, class_report
