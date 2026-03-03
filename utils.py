import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def get_benchmark(model, df: pd.DataFrame) -> tuple[float, float, str]:
    df = df.copy()

    df['Credit_Score'] = df['Credit_Score'].map({
        'Poor': 0,
        'Standard': 1,
        'Good': 2
    })

    target = 'Credit_Score'
    y = df[target]

    X = df.drop(columns=["Customer_ID", "ID", "Name", "SSN", "Month", target])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    class_report = classification_report(y_test, y_pred)

    return acc, auc, class_report


def data_splitter(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data for training and testing.
        target_col (str): The name of the target column in the DataFrame.

    Returns:
        tuple: A tuple containing the training features, testing features, training labels, and testing labels.
    """
    df = df.copy()

    df[target] = df[target].map({
        'Poor': 0,
        'Standard': 1,
        'Good': 2
    })

    target = 'Credit_Score'
    y = df[target]

    # Important features (exploration on features_selection.ipynb)
    important_features = ['Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date', 'Num_Credit_Card',
                         'Credit_Mix_Standard', 'Changed_Credit_Limit',
                          'Credit_Mix_Good', 'Payment_of_Min_Amount_Yes', 'High_spent']

    # droping non-informative and target columns
    X = df.drop(columns=["Customer_ID", "ID", "Name", "SSN", "Month", target])
    # keeping only important features for class differentiation
    X = X.drop(
        columns=[col for col in X.columns if col not in important_features])

    # Negate features for coherent score direction (higher score = better customer).
    negative_features = ['Interest_Rate', 'Outstanding_Debt', 'Delay_from_due_date',
                         'Payment_of_Min_Amount_Yes', 'Credit_Mix_Standard', 'Num_Credit_Card', 'Changed_Credit_Limit']
    X[negative_features] = X[negative_features] * -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
