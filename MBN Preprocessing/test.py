"""
test.py — Lightweight pipeline smoke-test (Steps 1–4 only).

Filenames use the same stage-prefixed, timestamped convention as main.py.
Adjust FILE_PATH and PROTECT_COLS as needed before running.
"""

from datetime import datetime
from heatmap import run_missingness_pipeline
from spearman import run_spearman_pipeline
from VIF import run_vif_pipeline
from outliers import run_outlier_pipeline

# =========================
# Configuration
# =========================
FILE_PATH    = "Full database.xlsx"
PROTECT_COLS = ["Paper"]

timestamp = datetime.now().strftime("%m.%d %H-%M")


def dated(stage, label, ext="xlsx"):
    """Return a step-labelled, timestamped filename."""
    return f"Step {stage} — {label} ({timestamp}).{ext}"


# =========================
# Pipeline (Steps 1-4)
# =========================
if __name__ == "__main__":

    # --- Step 1: Missingness + Encoding + Imputation ---
    print("\n" + "=" * 50)
    print("STEP 1: Missingness Heatmap, Encoding & Imputation")
    print("=" * 50)
    df_cleaned, encoded_cols = run_missingness_pipeline(
        file_path=FILE_PATH,
        output_path=dated("01", "Cleaned Dataset")
    )
    print(f"\nEncoded columns carried forward: {encoded_cols}")

    # --- Step 2: Spearman Correlation ---
    print("\n" + "=" * 50)
    print("STEP 2: Spearman Correlation")
    print("=" * 50)
    df_reduced = run_spearman_pipeline(
        df=df_cleaned,
        encoded_cols=encoded_cols,
        output_path=dated("02", "Reduced After Correlation"),
        threshold=0.8
    )

    # --- Step 3: VIF ---
    print("\n" + "=" * 50)
    print("STEP 3: VIF & Multicollinearity Filtering")
    print("=" * 50)
    df_multicol, dropped_vif = run_vif_pipeline(
        df=df_reduced,
        threshold=5,
        protect_cols=PROTECT_COLS,
        encoded_cols=encoded_cols,
        vif_output_path=dated("03", "VIF Results"),
        dropped_output_path=dated("03", "Dropped VIF Columns"),
        final_output_path=dated("04", "Dataset After VIF")
    )

    # --- Step 4: Outlier Removal ---
    print("\n" + "=" * 50)
    print("STEP 4: Outlier Detection & Removal")
    print("=" * 50)
    df_final, removed_rows, outlier_summary = run_outlier_pipeline(
        df=df_multicol,
        encoded_cols=encoded_cols,
        threshold=3,
        save_plots=True,
        output_path=dated("05", "Final No Outliers"),
        removed_output_path=dated("05", "Removed Outliers"),
        report_path=dated("05", "Outliers Report", ext="txt")
    )

    print("\n" + "=" * 50)
    print("TEST RUN COMPLETE (Steps 1-4)")
    print("=" * 50)
    print(f"  Rows removed by outlier step : {outlier_summary['rows_removed']}")