import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    """
    Plots a confusion matrix heatmap.

    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Engaged", "Frustrated"],
                yticklabels=["Engaged", "Frustrated"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_heatmap(true, recovered, title):
    """
    Plots a heatmap of the absolute difference between true and recovered matrices.

    """
    diff = np.abs(true - recovered)
    plt.figure(figsize=(6, 5))
    sns.heatmap(diff, annot=True, cmap="Reds", fmt=".2f", cbar=True)
    plt.title(f"Heatmap of abs(True - Recovered): {title}")
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def summarize_and_plot(results, csv_path="Logs/summary.csv"):
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["Sample Size"], df["Accuracy"], marker='o', label="Accuracy")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Accuracy")
    plt.title("HMM Accuracy vs. Sample Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df




def plot_kld_state_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.plot(df["Sample Size"], df["KLD_StateDist"], marker='o', color='orange', label="KL Divergence (State Dist.)")
    plt.xlabel("Sample Size")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence of Predicted vs. True State Distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_accuracy_per_seed(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="sample_size", y="accuracy", hue="seed", marker="o")
    plt.title("Accuracy vs Sample Size (per seed)")
    plt.xlabel("Sample Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend(title="Seed", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_kld_per_seed(df, kld_column, title):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="sample_size", y=kld_column, hue="seed", marker="o")
    plt.title(title)
    plt.xlabel("Sample Size")
    plt.ylabel("KL Divergence")
    plt.grid(True)
    plt.legend(title="Seed", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
