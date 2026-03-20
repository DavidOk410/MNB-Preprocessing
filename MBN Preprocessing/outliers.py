import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import zscore


def remove_outliers_zscore(df, threshold=3, exclude_cols=None):
    """
    Remove rows where any *original* numeric column has |Z-score| > threshold.

    Encoded (one-hot) columns are excluded from the Z-score calculation so that
    binary 0/1 flags do not incorrectly flag rows as outliers — but the encoded
    columns are kept intact in the returned dataframe.

    Parameters:
        df (pd.DataFrame)
        threshold (float)
        exclude_cols (list): one-hot encoded column names to skip during scoring

    Returns:
        df_clean (pd.DataFrame)
        summary (dict)
    """
    if exclude_cols is None:
        exclude_cols = []

    # Only score on original numeric columns (not binary encoded ones)
    score_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    z_scores = np.abs(zscore(df[score_cols], nan_policy='omit'))
    mask = (z_scores < threshold).all(axis=1)

    rows_before = df.shape[0]
    df_clean = df[mask]          # encoded columns ride along automatically
    rows_after = df_clean.shape[0]

    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after
    }

    print("\nOutlier Removal Summary:")
    print(f"  Scored on    : {len(score_cols)} original numeric columns")
    print(f"  Skipped      : {len(exclude_cols)} encoded columns")
    print(f"  Rows before  : {rows_before}")
    print(f"  Rows after   : {rows_after}")
    print(f"  Rows removed : {summary['rows_removed']}")

    return df_clean, summary


def qq_plots(df, exclude_cols=None, save=True, folder="qqplots", threshold=3):
    """
    Generate Q-Q plots for original numeric columns (encoded columns skipped).
    Dots whose |Z-score| > threshold are highlighted in red; normal dots in blue.

    Parameters:
        df (pd.DataFrame)
        exclude_cols (list): one-hot encoded column names to skip
        save (bool)
        folder (str)
        threshold (float): Z-score cutoff used to colour outlier dots red
    """
    if exclude_cols is None:
        exclude_cols = []

    os.makedirs(folder, exist_ok=True)

    plot_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    for col in plot_cols:
        series = df[col].dropna().reset_index(drop=True)

        # Compute theoretical quantiles and sorted sample values
        (osm, osr), (slope, intercept, _) = stats.probplot(series, dist="norm")

        # Z-scores aligned to the sorted order that probplot uses
        col_zscore = np.abs(zscore(series, nan_policy='omit'))
        sorted_zscores = col_zscore[np.argsort(series)]
        is_outlier = sorted_zscores > threshold

        fig, ax = plt.subplots(figsize=(5, 5))

        # Normal dots — blue
        ax.scatter(osm[~is_outlier], osr[~is_outlier],
                   color="steelblue", s=20, zorder=3, label="Normal")

        # Outlier dots — red
        if is_outlier.any():
            ax.scatter(osm[is_outlier], osr[is_outlier],
                       color="red", s=30, zorder=4, label=f"Outlier (|Z| > {threshold})")

        # Reference line
        x_line = np.array([osm.min(), osm.max()])
        ax.plot(x_line, slope * x_line + intercept,
                color="black", linewidth=1, zorder=2)

        n_outliers = int(is_outlier.sum())
        ax.set_title(f"Q-Q Plot: {col}\n({n_outliers} outlier(s) highlighted)", fontsize=9)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        if is_outlier.any():
            ax.legend(fontsize=7)

        plt.tight_layout()

        if save:
            safe_col = "".join(
                c if c.isalnum() or c in (" ", "_", "-") else "_" for c in col
            ).strip()
            file_path = os.path.join(folder, f"qqplot_{safe_col}.png")
            plt.savefig(file_path, dpi=300)

        plt.close()

    print(f"\nQ-Q plots generated for {len(plot_cols)} variables")
    if save:
        print(f"Plots saved in '{folder}/'")


def run_outlier_pipeline(
    df,
    encoded_cols=None,
    threshold=3,
    save_plots=True,
    output_path="Final_dataset_No_outliers.xlsx"
):
    """
    Full outlier pipeline:
      1. Q-Q plots on original numeric columns (encoded cols skipped),
         with outlier dots highlighted in red
      2. Z-score outlier removal based on original numeric columns only
         (encoded columns are kept in the returned dataframe unchanged)
      3. Save final dataset

    Returns:
        df_clean (pd.DataFrame): outlier-free dataframe with encoded cols intact
        summary (dict)
    """
    print("\n--- Generating Q-Q plots (original numeric columns only) ---")
    qq_plots(df, exclude_cols=encoded_cols, save=save_plots, threshold=threshold)

    print(f"\n--- Removing outliers (Z-score threshold: {threshold}) ---")
    df_clean, summary = remove_outliers_zscore(df, threshold=threshold, exclude_cols=encoded_cols)

    df_clean.to_excel(output_path, index=False)
    print(f"\nFinal dataset saved as '{output_path}'")
    print("Final shape:", df_clean.shape)

    return df_clean, summary