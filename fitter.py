import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils import data_splitter


def benchmark_fitter():
    """
    Fit a simple logistic regression model on the raw features to establish a benchmark.
    """
    df = pd.read_csv("Data/clean_train.csv")

    df['Credit_Score'] = df['Credit_Score'].map({
        'Poor': 0,
        'Standard': 1,
        'Good': 2
    })

    target = 'Credit_Score'
    y = df[target]

    X = df.drop(columns=["Customer_ID", "ID", "Name", "SSN", "Month", target])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, 'Models/benchmark_model.pkl')


def weights_model_fitter():
    """
    Fit a logistic regression model on the WoE-encoded training data to get feature weights.
    """
    df = pd.read_csv("Data/clean_train.csv")

    X_train, _, y_train, _ = data_splitter(df, target='Credit_Score')

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, 'Models/weights_model.pkl')


def main():
    benchmark_fitter()
    weights_model_fitter()


if __name__ == "__main__":
    main()
