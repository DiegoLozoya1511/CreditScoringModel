import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

blue_colors = [
    "#D9EDFF",
    "#C1E2FF",
    "#A8D5FF",
    "#8AC6FF",
    "#6BB6FF",
    "#4A9EE5",
    "#2E86DE",
    "#0066C0",
    "#0055A5",
    "#003D82"
]

blue_scale = mcolors.LinearSegmentedColormap.from_list(
    "blue_scale",
    [blue_colors[0], blue_colors[-1]]
)


def plot_score_distribution(df: pd.DataFrame, tresholds: list, set: str):
    """
    Plot the distribution of the custom credit score.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Score' column with credit scores.
        tresholds (list): List of treshold values used for classifying credit scores.
        set (str): Name of the dataset (e.g., 'Train' or 'Test') for the plot title.
    """
    plt.figure()
    sns.kdeplot(df['Score'], color=blue_colors[-1],
                fill=True, alpha=0.3, linewidth=2)
    plt.title(f"{set} Distribution of Custom Credit Score")
    plt.xlabel("Custom Credit Score")
    plt.ylabel("Density")
    plt.show()

    plt.figure()
    color_idx = [1, 5, 9]
    labels = {0: 'Poor', 1: 'Standard', 2: 'Good'}

    # Plot Tresholds
    plt.axvline(x=tresholds[0], color='dimgray', linestyle='--',
                alpha=1.0, label='Poor/Standard Tresholds')
    plt.axvline(x=tresholds[1], color='black', linestyle='--',
                alpha=0.7, label='Standard/Good Tresholds')

    for class_idx in range(3):
        sns.kdeplot(
            df[df['Credit_Score'] == class_idx]['Score'],
            color=blue_colors[color_idx[class_idx]],
            fill=True,
            alpha=0.3,
            linewidth=2,
            label=labels[class_idx]
        )

    plt.title(f"{set} Score Distribution per Class")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def plot_score_comparison(scores: pd.DataFrame, set: str):
    """
    Compares the actual credit score distribution with the classified score distribution.

    Parameters:
        scores (pd.DataFrame): DataFrame containing 'Credit_Score' and 'Score' columns.
        set (str): Name of the dataset (e.g., 'Train' or 'Test') for the plot title.
    """
    real_classes = [
        scores[scores['Credit_Score'] == 0]['Score'],
        scores[scores['Credit_Score'] == 1]['Score'],
        scores[scores['Credit_Score'] == 2]['Score']
    ]

    classified_classes = [
        scores[scores['Class'] == 0]['Score'],
        scores[scores['Class'] == 1]['Score'],
        scores[scores['Class'] == 2]['Score']
    ]

    class_labels = ["Poor", "Standard", "Good"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for i in range(3):

        sns.kdeplot(
            real_classes[i],
            color=blue_colors[2],
            fill=True,
            alpha=0.3,
            linewidth=2,
            label=f"Real {class_labels[i]}",
            ax=axes[i]
        )

        sns.kdeplot(
            classified_classes[i],
            color=blue_colors[-1],
            fill=True,
            alpha=0.3,
            linewidth=2,
            label=f"Classified {class_labels[i]}",
            ax=axes[i]
        )

        axes[i].set_title(f"{class_labels[i]}")
        axes[i].set_xlabel("Score")

        axes[i].legend()

    axes[0].set_ylabel("Density")
    fig.suptitle(f"{set} Real vs Classified Score Distribution", fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(scores: pd.DataFrame, set: str) -> str:
    """
    Plot the confusion matrix for the given true and predicted labels.

    Parameters:
        scores (pd.DataFrame): DataFrame containing 'Credit_Score' (true labels) and 'Class' (predicted labels).
        set (str): Name of the dataset (e.g., 'Train' or 'Test') for the plot title.

    Returns:
        class_report (str): The classification report as a string.
    """
    acc = accuracy_score(scores['Credit_Score'], scores['Class'])
    class_report = classification_report(
        scores['Credit_Score'], scores['Class'], target_names=['Poor', 'Standard', 'Good'])

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay.from_predictions(
        scores['Credit_Score'],
        scores['Class'],
        cmap=blue_scale,
        colorbar=True,
        values_format='d',
        ax=ax
    )

    ax.set_title(f'{set} Confusion Matrix - Accuracy: {acc:.4f}')
    ax.grid(False)

    plt.show()

    return class_report
