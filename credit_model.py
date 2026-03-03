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
        'feature'    : features,
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
    X_train_woe = X_train_woe[weights.T.columns]  # Ensure feature order matches weights
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