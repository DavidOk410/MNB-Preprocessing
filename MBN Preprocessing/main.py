from heatmap import run_missingness_pipeline
from spearman import run_spearman_pipeline
from VIF import run_vif_pipeline
from outliers import run_outlier_pipeline
from stats_summary import run_stats_pipeline
from colorize import colorize_binary_columns

# =========================
# Configuration
# =========================
FILE_PATH = "Full database.xlsx"   # <-- change this to your input file

# Columns that should never be dropped during VIF removal (e.g. ID or target columns)
PROTECT_COLS = ["Paper"]           # <-- adjust as needed

# =========================
# Pipeline
# =========================
if __name__ == "__main__":

    # --- Step 1: Missingness + Encoding + Imputation ---
    print("\n" + "=" * 50)
    print("STEP 1: Missingness Heatmap, Encoding & Imputation")
    print("=" * 50)
    df_cleaned, encoded_cols = run_missingness_pipeline(
        file_path=FILE_PATH,
        output_path="Cleaned_dataset Task 2.xlsx"
    )
    print(f"\nEncoded columns carried forward: {encoded_cols}")

    # --- Step 2: Spearman Correlation (encoded cols excluded) ---
    print("\n" + "=" * 50)
    print("STEP 2: Spearman Correlation")
    print("=" * 50)
    df_reduced = run_spearman_pipeline(
        df=df_cleaned,
        encoded_cols=encoded_cols,
        output_path="Reduced_dataset_after_correlation.xlsx",
        threshold=0.8
    )

    # --- Step 3: VIF (encoded cols excluded, threshold=5) ---
    print("\n" + "=" * 50)
    print("STEP 3: VIF & Multicollinearity Filtering")
    print("=" * 50)
    df_multicol, dropped_vif = run_vif_pipeline(
        df=df_reduced,
        threshold=5,
        protect_cols=PROTECT_COLS,
        encoded_cols=encoded_cols,
        vif_output_path="VIF_results.xlsx",
        dropped_output_path="Dropped_VIF_columns.xlsx",
        final_output_path="Dataset_Multicol.xlsx"
    )

    # --- Step 4: Outlier Removal (encoded cols excluded from Z-score) ---
    print("\n" + "=" * 50)
    print("STEP 4: Outlier Detection & Removal")
    print("=" * 50)
    df_final, outlier_summary = run_outlier_pipeline(
        df=df_multicol,
        encoded_cols=encoded_cols,
        threshold=3,
        save_plots=True,
        output_path="Final_dataset_No_outliers.xlsx"
    )

    # --- Step 4.1: Colorize categorical ---
    print("\n" + "=" * 50)
    print("Step 4.1: Colorize categorical")
    print("=" * 50)
    colorize_binary_columns("Final_dataset_No_outliers.xlsx")

    # --- Step 5: Statistical Feature Characteristics ---
    print("\n" + "=" * 50)
    print("STEP 5: Statistical Feature Characteristics")
    print("=" * 50)
    stats_summary = run_stats_pipeline(
        df=df_final,
        exclude_cols=encoded_cols,
        output_path="Statistical_summary.xlsx"
    )



    # --- Summary ---
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Input shape              : {df_cleaned.shape}")
    print(f"  After correlation filter : {df_reduced.shape}")
    print(f"  After VIF filter         : {df_multicol.shape}")
    print(f"  After outlier removal    : {df_final.shape}")
    print(f"  Outlier rows removed     : {outlier_summary['rows_removed']}")
    print(f"  Encoded columns (kept)   : {encoded_cols}")
    print("\nOutput files:")
    print("  - Cleaned_dataset Task 2.xlsx")
    print("  - Reduced_dataset_after_correlation.xlsx")
    print("  - VIF_results.xlsx")
    print("  - Dropped_VIF_columns.xlsx")
    print("  - Dataset_Multicol.xlsx")
    print("  - Final_dataset_No_outliers.xlsx")
    print("  - Statistical_summary.xlsx")
    print("  - missingness_heatmap.png")
    print("  - spearman_heatmap.png")
    print("  - qqplots/")