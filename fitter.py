import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils import data_splitter
from WoE_encoding import get_woe_encoded_data


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


def get_woe_dfs():
    """
    Fit WoE encoding on the training data and transform both train and test sets.
    """
    df = pd.read_csv("Data/clean_train.csv")
    
    X_train, X_test, y_train, _ = data_splitter(df, target='Credit_Score')
    
    features = X_train.columns.tolist()
    X_train_woe, X_test_woe, iv_summary = get_woe_encoded_data(X_train, y_train, X_test, features)
    
    X_train_woe.to_csv('Data/X_train_woe.csv', index=False)
    X_test_woe.to_csv('Data/X_test_woe.csv', index=False)
    

def weights_model_fitter():
    """
    Fit a logistic regression model on the WoE-encoded training data to get feature weights.
    """
    df = pd.read_csv("Data/clean_train.csv")
    
    _, _, y_train, _ = data_splitter(df, target='Credit_Score')
    
    X_train_woe = pd.read_csv('Data/X_train_woe.csv')
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train_woe, y_train)

    joblib.dump(model, 'Models/weights_model.pkl')


def main():
    benchmark_fitter()
    get_woe_dfs()
    weights_model_fitter()

 
if __name__ == "__main__":
    main()