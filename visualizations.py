from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

def plot_score_distribution(df: pd.DataFrame, set: str):
    """
    Plot the distribution of the custom credit score.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Score' column with credit scores.
    """
    plt.figure()
    sns.kdeplot(df['Score'], color=blue_colors[-1], fill=True, alpha=0.3, linewidth=2)
    plt.title(f"{set} Distribution of Custom Credit Score")
    plt.xlabel("Custom Credit Score")
    plt.ylabel("Density")
    plt.show()
    
    plt.figure()
    color_idx = [1, 5, 9]
    labels    = {0: 'Poor', 1: 'Standard', 2: 'Good'}

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