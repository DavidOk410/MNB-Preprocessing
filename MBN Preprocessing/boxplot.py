import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def run_boxplot_pipeline(
    df,
    exclude_cols=None,
    output_path="Scaled_dataset.xlsx",
    plot_path="boxplots_scaled.png"
):
    """
    Apply Standard Scaling to continuous (original numeric) features,
    generate a boxplot similar to Fig. 5, and save the scaled dataset.

    Parameters:
        df (pd.DataFrame): cleaned dataset after outlier removal
        exclude_cols (list): one-hot encoded / ID columns to skip
        output_path (str): path to save the scaled Excel dataset
        plot_path (str): path to save the boxplot image

    Returns:
        df_scaled (pd.DataFrame): dataset with scaled continuous features
        scaler (StandardScaler): fitted scaler
    """
    if exclude_cols is None:
        exclude_cols = []

    # Identify continuous (original numeric) columns
    continuous_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # ============================================
    # Apply Standard Scaling to continuous features only
    # ============================================
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[continuous_cols])
    df_scaled_cont = pd.DataFrame(scaled_features, columns=continuous_cols, index=df.index)

    # Rebuild full dataframe: scaled continuous + untouched encoded cols
    df_scaled = df.copy()
    df_scaled[continuous_cols] = df_scaled_cont

    # ============================================
    # Create boxplot similar to Figure 5 in the paper
    # ============================================
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_scaled_cont, palette="Set2",
                flierprops=dict(marker="o", markersize=3,
                                markerfacecolor="red", markeredgecolor="red",
                                alpha=0.6))

    plt.xticks(rotation=45, ha='right')
    plt.title("Standardized features after preprocessing (similar to Fig. 5)",
              fontsize=14, fontweight='bold')
    plt.ylabel("Standardized value", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.show()

    print("\n✅ Boxplot generated successfully")
    print(f"   Saved as '{plot_path}'")

    # ============================================
    # Save scaled dataset
    # ============================================
    df_scaled.to_excel(output_path, index=False)
    print(f"✅ Scaled dataset saved as '{output_path}'")
    print(f"   Shape: {df_scaled.shape}")

    return df_scaled, scaler


# ============================================
# Run standalone (outside main pipeline)
# ============================================
if __name__ == "__main__":
    INPUT_FILE   = "Final_dataset_No_outliers.xlsx"   # <-- change if needed
    EXCLUDE_COLS = []                                  # <-- add encoded col names if needed

    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded dataset: {df.shape}")

    run_boxplot_pipeline(
        df=df,
        exclude_cols=EXCLUDE_COLS,
        output_path="Scaled_dataset.xlsx",
        plot_path="boxplots_scaled.png"
    )