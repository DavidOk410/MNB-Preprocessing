from heatmap import run_missingness_pipeline
from spearman import run_spearman_pipeline
from VIF import run_vif_pipeline
from outliers import run_outlier_pipeline

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
    df_final, dropped_vif = run_vif_pipeline(
        df=df_reduced,
        threshold=5,
        protect_cols=PROTECT_COLS,
        encoded_cols=encoded_cols,
        vif_output_path="VIF_results.xlsx",
        dropped_output_path="Dropped_VIF_columns.xlsx",
        final_output_path="Dataset_Multicol.xlsx"
    )