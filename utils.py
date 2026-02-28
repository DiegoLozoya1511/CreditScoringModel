import pandas as pd
from sklearn.model_selection import train_test_split


def data_splitter(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data for training and testing.
        target_col (str): The name of the target column in the DataFrame.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Controls the randomness of the split (default is 42).

    Returns:
        tuple: A tuple containing the training features, testing features, training labels, and testing labels.
    """
    df = df.copy()

    df[target] = df[target].map({
        'Poor': 0,
        'Standard': 1,
        'Good': 2
    })

    # Important features (exploration on features_selection.ipynb)
    important_features = ['Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date', 'Num_Credit_Card',
                          'Num_of_Delayed_Payment', 'High_spent', 'Credit_Mix_Standard', 'Changed_Credit_Limit',
                          'Credit_Mix_Good', 'Payment_of_Min_Amount_Yes', 'Total_EMI_per_month']

    # droping non-informative and target columns
    X = df.drop(columns=["Customer_ID", "ID", "Name", "SSN", "Month", target])
    # keeping only important features for class differentiation
    X = X.drop(
        columns=[col for col in X.columns if col not in important_features])

    target = 'Credit_Score'
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
