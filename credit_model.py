import pandas as pd


def get_weights(model, features: list) -> pd.DataFrame:
    """
    Extract feature weights from a fitted model and return them in a DataFrame.

    Parameters:
        model (sklearn estimator): A fitted sklearn model with coef_ attribute.
        features (list): List of feature names corresponding to the model's coefficients.

    Returns:
        pd.DataFrame: DataFrame containing features and their corresponding coefficients,
                      sorted by coefficient value in descending order.
    """
    weights = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_[2]  # class 2 = Good
    }).sort_values('coefficient', ascending=False).reset_index(drop=True)

    return weights


def get_scores(weights: pd.DataFrame, X_train_woe: pd.DataFrame) -> pd.Series:
    """
    Calculate credit scores based on feature weights and input features.

    Parameters:
        weights (pd.DataFrame): DataFrame containing features and their coefficients.
        features (pd.DataFrame): DataFrame containing the feature values for each customer.

    Returns:
        pd.Series: Series containing the calculated credit score for each customer.
    """
    weights = weights.set_index('feature')
    # Ensure feature order matches weights
    X_train_woe = X_train_woe[weights.T.columns]
    scores = X_train_woe.dot(weights['coefficient'])
    return scores


def scale_scores(scores: pd.Series, min_value: float) -> pd.Series:
    """
    Scale raw logistic regression scores to a fixed integer range.

    Shifts scores to be non-negative, scales by 100, then clips
    and normalizes between 0 - 1000.

    Parameters:
        scores    (pd.Series) : Raw scores from the logistic regression.
        min_value (float)     : Minimum value used to shift scores non-negative.

    Returns:
        scaled (pd.Series): Integer scores constrained between score_min and score_max.
    """
    score_min = 0
    score_max = 600

    shifted = (scores + abs(min_value)) * 100

    # Normalize to [score_min, score_max]
    scaled = (shifted - shifted.min()) / (shifted.max() - shifted.min())
    scaled = (scaled * (score_max - score_min) + score_min).astype(int)

    return scaled.clip(score_min, score_max)


def scores_df(weights: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate and scale credit scores for both training and testing datasets.

    Parameters:
        weights (pd.DataFrame): DataFrame containing features and their coefficients.
        X_train (pd.DataFrame): Training features (WoE-encoded).
        y_train (pd.Series): Training labels.
        X_test  (pd.DataFrame): Testing features (WoE-encoded).
        y_test  (pd.Series): Testing labels.

    Returns:
        tuple: A tuple containing two DataFrames with original credit scores and scaled scores for train and test sets.
    """
    # - Train -
    scores = get_scores(weights, X_train)
    min_value = scores.min()  # Save min train value for consistent scaling of test scores
    train_scores = pd.DataFrame({
        'Credit_Score': y_train,
        'Score': scale_scores(scores, min_value)
    }).reset_index(drop=True)

    # - Test -
    scores = get_scores(weights, X_test)  # X_test_woe
    test_scores = pd.DataFrame({
        'Credit_Score': y_test,
        'Score': scale_scores(scores, min_value)
    }).reset_index(drop=True)

    return train_scores, test_scores


def treshold_classification(train_scores: pd.DataFrame, test_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Classify customers into credit score classes based on defined tresholds.

    Parameters:
        train_scores (pd.DataFrame): DataFrame containing 'Credit_Score' and 'Score' for the training set.
        test_scores  (pd.DataFrame): DataFrame containing 'Credit_Score' and 'Score' for the testing set.

    Returns:
        tuple: A tuple containing the modified train_scores and test_scores DataFrames with an added 'Class' column, and the list of tresholds used for classification.
    """
    # Tresholds Definition (arbitraryly set based on score distribution and class separation)
    t1 = 220  # Poor vs Standard
    t2 = 410  # Standard vs Good
    tresholds = [t1, t2]

    # Classify based on tresholds
    train_scores['Class'] = train_scores['Score'].apply(
        lambda x: 0 if x <= t1 else 1 if x < t2 else 2)
    test_scores['Class'] = test_scores['Score'].apply(
        lambda x: 0 if x <= t1 else 1 if x < t2 else 2)

    return train_scores, test_scores, tresholds
