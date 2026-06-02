"""
boxplot.py — Standard scaling and boxplot visualisation (Step 6 of the pipeline).

Enhancements:
  - Horizontal dashed lines at ±3σ (always at y = ±3 on standardised data).
  - Each box is annotated with the count of data points outside ±3σ.
  - Plot filename follows the same timestamped naming convention as other outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def run_boxplot_pipeline(
    df,
    exclude_cols=None,
    output_path="06_scaled_dataset.xlsx",
    plot_path="boxplots_scaled.png"
):
    """
    Apply Standard Scaling to continuous (original numeric) features,
    generate a boxplot of the scaled features with ±3σ threshold lines
    and per-feature outlier counts, and save the scaled dataset.

    One-hot encoded columns (passed via `exclude_cols`) are preserved in the
    output dataset but are NOT scaled or plotted.

    Parameters
    ----------
    df           : pd.DataFrame — cleaned dataset after outlier removal
    exclude_cols : list         — one-hot encoded / ID columns to skip
    output_path  : str          — path to save the scaled Excel dataset
    plot_path    : str          — path to save the boxplot image

    Returns
    -------
    df_scaled : pd.DataFrame   — dataset with scaled continuous features
    scaler    : StandardScaler — fitted scaler (for downstream use if needed)
    """
    if exclude_cols is None:
        exclude_cols = []

    continuous_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # ----------------------------------------------------------------
    # Standard scaling (continuous features only)
    # ----------------------------------------------------------------
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[continuous_cols])
    df_scaled_cont = pd.DataFrame(scaled_features, columns=continuous_cols,
                                  index=df.index)

    # Rebuild full dataframe: scaled continuous + untouched encoded cols
    df_scaled = df.copy()
    df_scaled[continuous_cols] = df_scaled_cont

    # ----------------------------------------------------------------
    # Per-feature outlier counts (|z| > 3 on standardised data)
    # ----------------------------------------------------------------
    outlier_counts = (df_scaled_cont.abs() > 3).sum(axis=0)

    # ----------------------------------------------------------------
    # Boxplot with ±3σ lines and outlier annotations
    # ----------------------------------------------------------------
    n_features = len(continuous_cols)
    fig_width = max(12, n_features * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    sns.boxplot(
        data=df_scaled_cont,
        palette="Set2",
        flierprops=dict(
            marker="o", markersize=3,
            markerfacecolor="red", markeredgecolor="red",
            alpha=0.6
        ),
        ax=ax
    )

    # ±3σ horizontal threshold lines
    ax.axhline(y= 3, color="crimson", linestyle="--", linewidth=1.2,
               label="±3σ threshold")
    ax.axhline(y=-3, color="crimson", linestyle="--", linewidth=1.2)

    # Annotate each box with outlier count
    x_positions = range(len(continuous_cols))
    y_max = df_scaled_cont.max().max()
    annotation_y = max(3.3, y_max + 0.2)

    for x_pos, col in zip(x_positions, continuous_cols):
        count = int(outlier_counts[col])
        if count > 0:
            ax.text(
                x_pos, annotation_y, f"n={count}",
                ha="center", va="bottom", fontsize=7,
                color="crimson", fontweight="bold"
            )

    ax.set_xticks(range(len(continuous_cols)))
    ax.set_xticklabels(continuous_cols, rotation=45, ha='right')
    ax.set_title(
        "Standardised features after preprocessing — boxplot with ±3σ thresholds",
        fontsize=13, fontweight='bold'
    )
    ax.set_ylabel("Standardised value", fontsize=12)
    ax.set_xlabel("Features", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"  Boxplot saved         : {plot_path}")
    print(f"  Features with |z|>3   : {outlier_counts[outlier_counts > 0].to_dict()}")

    # ----------------------------------------------------------------
    # Save scaled dataset
    # ----------------------------------------------------------------
    df_scaled.to_excel(output_path, index=False)
    print(f"  Scaled dataset saved  : {output_path}")
    print(f"  Shape                 : {df_scaled.shape}")

    return df_scaled, scaler


# ----------------------------------------------------------------
# Standalone entry point
# ----------------------------------------------------------------
if __name__ == "__main__":
    INPUT_FILE   = "05_final_no_outliers.xlsx"
    EXCLUDE_COLS = []

    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded dataset: {df.shape}")

    run_boxplot_pipeline(
        df=df,
        exclude_cols=EXCLUDE_COLS,
        output_path="06_scaled_dataset.xlsx",
        plot_path="boxplots_scaled.png"
    )
