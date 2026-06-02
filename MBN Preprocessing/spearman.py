import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_spearman_correlation(df, exclude_cols=None):
    """
    Compute Spearman correlation matrix on numeric columns,
    excluding one-hot encoded columns passed via `exclude_cols`.

    Parameters:
        df (pd.DataFrame)
        exclude_cols (list): column names to skip (e.g. one-hot encoded columns)

    Returns:
        corr_matrix (pd.DataFrame)
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_df = (
        df.select_dtypes(include=[np.number])
          .drop(columns=exclude_cols, errors='ignore')
    )

    print(f"\nComputing Spearman correlation on {len(numeric_df.columns)} columns "
          f"(excluded {len(exclude_cols)} encoded columns).")

    corr_matrix = numeric_df.corr(method='spearman')
    return corr_matrix


def plot_spearman_heatmap(corr_matrix, save_path="spearman_heatmap.png"):
    """Generate and save the Spearman correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")

    plt.title("Spearman Correlation Heatmap", fontsize=12)
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSpearman heatmap saved as '{save_path}'")


def drop_high_correlation_cols(df, corr_matrix, threshold=0.8):
    """
    Drop one feature from each highly correlated pair (|R| > threshold).
    Only considers columns present in corr_matrix — encoded columns are untouched.
    """
    corr_matrix_abs = corr_matrix.abs()

    upper = corr_matrix_abs.where(
        np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"\nHighly correlated columns to drop (|R| > {threshold}):")
    print(to_drop)

    df_reduced = df.drop(columns=to_drop)
    print("Shape after correlation filtering:", df_reduced.shape)

    return df_reduced


def run_spearman_pipeline(df, encoded_cols=None, output_path="Reduced_dataset_after_correlation.xlsx", threshold=0.8, heatmap_path="spearman_heatmap.png"):
    """
    Full Spearman correlation pipeline:
      1. Compute Spearman correlation (ignoring encoded columns)
      2. Plot and save heatmap
      3. Drop highly correlated features (encoded columns are never dropped)
      4. Save reduced dataset

    Parameters:
        df (pd.DataFrame)
        encoded_cols (list): one-hot column names to exclude from analysis
        output_path (str)
        threshold (float)
        heatmap_path (str): path/filename for the saved heatmap image

    Returns:
        df_reduced (pd.DataFrame)
    """
    corr_matrix = compute_spearman_correlation(df, exclude_cols=encoded_cols)
    plot_spearman_heatmap(corr_matrix, save_path=heatmap_path)
    df_reduced = drop_high_correlation_cols(df, corr_matrix, threshold=threshold)

    df_reduced.to_excel(output_path, index=False)
    print(f"\nReduced dataset saved as '{output_path}'")

    return df_reduced