"""
boxplot.py — Standard scaling and boxplot visualisation (Step 6 of the pipeline).

This module is integrated into main.py as the final processing stage.
It can also be run standalone for ad-hoc use.
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
    generate a boxplot of the scaled features, and save the scaled dataset.

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
    # Boxplot (mirrors Figure 5 style from the paper)
    # ----------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_scaled_cont,
        palette="Set2",
        flierprops=dict(
            marker="o", markersize=3,
            markerfacecolor="red", markeredgecolor="red",
            alpha=0.6
        )
    )

    plt.xticks(rotation=45, ha='right')
    plt.title("Standardized features after preprocessing (similar to Fig. 5)",
              fontsize=14, fontweight='bold')
    plt.ylabel("Standardized value", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"  Boxplot saved         : {plot_path}")

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
    INPUT_FILE   = "05_final_no_outliers.xlsx"   # adjust as needed
    EXCLUDE_COLS = []                             # add encoded col names if needed

    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded dataset: {df.shape}")

    run_boxplot_pipeline(
        df=df,
        exclude_cols=EXCLUDE_COLS,
        output_path="06_scaled_dataset.xlsx",
        plot_path="boxplots_scaled.png"
    )
