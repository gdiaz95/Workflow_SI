import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_marginal_original_vs_npgc(original, non_param, feature_name="Feature", save_path=None):

    # Data Preparation
    orig_series = pd.Series(original).dropna()
    npgc_series = pd.Series(non_param).dropna()

    sns.set_style("white")
    plt.figure(figsize=(12, 7))

    if pd.api.types.is_numeric_dtype(orig_series):

        # 1. Original: Light blue shaded
        sns.kdeplot(
            orig_series,
            fill=True,
            alpha=0.35,
            color="lightblue",
            label="Original",
            zorder=1
        )

        # 2. NPGC: Solid purple line
        sns.kdeplot(
            npgc_series,
            fill=False,
            linewidth=7,
            linestyle='-',
            color="purple",
            label="NPGC",
            zorder=2
        )

        # Styling
        ax = plt.gca()
        plt.xlabel(feature_name, fontsize=20, labelpad=10)
        plt.ylabel("Density", fontsize=20, labelpad=10)

        ax.tick_params(axis='x', labelsize=16)
        ax.set_yticks([])

        sns.despine(ax=ax, offset=0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='gray')

        plt.legend(fontsize=18, frameon=True, facecolor='white',
                   framealpha=0.9, loc='upper right')

    plt.subplots_adjust(bottom=0.2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()