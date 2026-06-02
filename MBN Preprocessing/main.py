import sys
from datetime import datetime
from pathlib import Path

from heatmap import run_missingness_pipeline
from spearman import run_spearman_pipeline
from VIF import run_vif_pipeline
from outliers import run_outlier_pipeline
from stats_summary import run_stats_pipeline
from colorize import colorize_binary_columns
from boxplot import run_boxplot_pipeline

# =========================
# Configuration
# =========================
FILE_PATH    = "Full database.xlsx"
PROTECT_COLS = ["Paper"]

# Timestamp shared across all output filenames: MM.DD HH-MM
timestamp = datetime.now().strftime("%m.%d %H-%M")


def dated(stage, label, ext="xlsx"):
    """
    Build a human-readable, timestamped output filename.

    Format: Step <stage> — <Label> (<MM.DD HH-MM>).<ext>
    Example: Step 01 — Cleaned Dataset (05.26 14-35).xlsx
    """
    return f"Step {stage} — {label} ({timestamp}).{ext}"


def dated_fig(label, ext="png"):
    """
    Build a timestamped filename for figure/plot outputs.

    Format: Fig — <Label> (<MM.DD HH-MM>).<ext>
    Example: Fig — Missingness Heatmap (05.26 14-35).png
    """
    return f"Fig — {label} ({timestamp}).{ext}"


# =========================
# Logger
# =========================
class Logger:
    """
    Tees every byte written to sys.stdout to both the terminal and a .txt
    log file simultaneously.
    """

    def __init__(self, path):
        self._terminal = sys.stdout
        self._file     = open(path, "w", encoding="utf-8")
        self._file.write(
            f"Pipeline Log\n"
            f"Started  :  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            + "=" * 60 + "\n\n"
        )
        self._file.flush()

    def start(self):
        sys.stdout = self

    def write(self, text):
        self._terminal.write(text)
        self._terminal.flush()
        self._file.write(text)
        self._file.flush()

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def __call__(self, *args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        self.write(sep.join(str(a) for a in args) + end)

    def close(self):
        sys.stdout = self._terminal
        self._file.write(
            "\n" + "=" * 60 + "\n"
            f"Finished :  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        self._file.close()


# =========================
# Cleanup helper
# =========================
def delete_previous_outputs(log):
    """
    Delete timestamped output files created by previous pipeline runs.
    Only files matching the stage-labelled naming pattern are removed.
    Input files are never touched.
    """
    output_patterns = [
        "Step 01 — Cleaned Dataset (*).xlsx",
        "Step 01 — Cleaned Dataset (Encoded) (*).xlsx",
        "Step 01 — Cleaned Dataset * (Encoded).xlsx",
        "Step 02 — Reduced After Correlation (*).xlsx",
        "Step 03 — VIF Results (*).xlsx",
        "Step 03 — Dropped VIF Columns (*).xlsx",
        "Step 04 — Dataset After VIF (*).xlsx",
        "Step 05 — Statistical Summary (*).xlsx",
        "Step 06 — Scaled Dataset (*).xlsx",
        "Step 07 — Final No Outliers (*).xlsx",
        "Step 07 — Removed Outliers (*).xlsx",
        "Step 07 — Outliers Report (*).txt",
        "Step 00 — Pipeline Log (*).txt",
        "Fig — *(*).png",
    ]

    deleted = []
    for pattern in output_patterns:
        for file in Path(".").glob(pattern):
            try:
                file.unlink()
                deleted.append(file.name)
            except Exception as e:
                log(f"Could not delete {file.name}: {e}")

    if deleted:
        log("\nDeleted previous output files:")
        for f in deleted:
            log(f"  - {f}")
    else:
        log("\nNo previous output files found.")


# =========================
# Pipeline
# =========================
if __name__ == "__main__":

    log_file = dated("00", "Pipeline Log", ext="txt")
    log = Logger(log_file)
    log.start()

    answer = input(
        "Should I delete output files of the previous runs? Type yes/no: "
    ).strip().lower()
    log(f"Delete previous outputs: {answer}")

    if answer in ["yes", "y"]:
        delete_previous_outputs(log)
    else:
        log("Previous output files were not deleted.")

    # --- Output filenames (step-labelled + timestamped) ---
    cleaned_file          = dated("01", "Cleaned Dataset")
    # encoded intermediate path is auto-derived inside heatmap.py
    # as "Step 01 — Cleaned Dataset (Encoded) (<timestamp>).xlsx"
    encoded_intermediate  = dated("01", "Cleaned Dataset (Encoded)")
    reduced_file          = dated("02", "Reduced After Correlation")
    vif_file              = dated("03", "VIF Results")
    dropped_vif_file      = dated("03", "Dropped VIF Columns")
    multicol_file         = dated("04", "Dataset After VIF")
    stats_file            = dated("05", "Statistical Summary")
    scaled_file           = dated("06", "Scaled Dataset")
    final_file            = dated("07", "Final No Outliers")
    removed_file          = dated("07", "Removed Outliers")
    outlier_report        = dated("07", "Outliers Report", ext="txt")

    # --- Figure filenames (timestamped) ---
    heatmap_fig    = dated_fig("Missingness Heatmap")
    spearman_fig   = dated_fig("Spearman Heatmap")
    boxplot_fig    = dated_fig("Boxplots Scaled")
    qqplot_folder  = f"qqplots ({timestamp})"

    # --- Step 1: Missingness + Encoding + Imputation ---
    log("\n" + "=" * 50)
    log("STEP 1: Missingness Heatmap, Encoding & Imputation")
    log("=" * 50)
    df_cleaned, encoded_cols = run_missingness_pipeline(
        file_path=FILE_PATH,
        output_path=cleaned_file,
        encoded_output_path=encoded_intermediate,
        heatmap_path=heatmap_fig,
    )
    log(f"\nEncoded columns carried forward: {encoded_cols}")

    # --- Step 2: Spearman Correlation ---
    log("\n" + "=" * 50)
    log("STEP 2: Spearman Correlation")
    log("=" * 50)
    df_reduced = run_spearman_pipeline(
        df=df_cleaned,
        encoded_cols=encoded_cols,
        output_path=reduced_file,
        heatmap_path=spearman_fig,
        threshold=0.8
    )

    # --- Step 3: VIF ---
    log("\n" + "=" * 50)
    log("STEP 3: VIF & Multicollinearity Filtering")
    log("=" * 50)
    df_multicol, dropped_vif = run_vif_pipeline(
        df=df_reduced,
        threshold=5,
        protect_cols=PROTECT_COLS,
        encoded_cols=encoded_cols,
        vif_output_path=vif_file,
        dropped_output_path=dropped_vif_file,
        final_output_path=multicol_file
    )

    # --- Step 4.1: Colorize categorical columns ---
    log("\n" + "=" * 50)
    log("STEP 4.1: Colorize Categorical Columns")
    log("=" * 50)
    colorize_binary_columns(multicol_file)

    # --- Step 5: Statistical Feature Characteristics ---
    log("\n" + "=" * 50)
    log("STEP 5: Statistical Feature Characteristics")
    log("=" * 50)
    stats_summary = run_stats_pipeline(
        df=df_multicol,
        exclude_cols=encoded_cols,
        output_path=stats_file
    )

    # --- Step 6: Standard Scaling + Boxplot ---
    log("\n" + "=" * 50)
    log("STEP 6: Standard Scaling & Boxplot")
    log("=" * 50)
    df_scaled, scaler = run_boxplot_pipeline(
        df=df_multicol,
        exclude_cols=encoded_cols,
        output_path=scaled_file,
        plot_path=boxplot_fig
    )

    # --- Step 7: Outlier Detection & Removal (FINAL step) ---
    log("\n" + "=" * 50)
    log("STEP 7: Outlier Detection & Removal")
    log("=" * 50)
    df_final, removed_rows, outlier_summary = run_outlier_pipeline(
        df=df_multicol,
        encoded_cols=encoded_cols,
        threshold=3,
        save_plots=True,
        output_path=final_file,
        removed_output_path=removed_file,
        report_path=outlier_report,
        qqplot_folder=qqplot_folder,
    )

    # --- Summary ---
    log("\n" + "=" * 50)
    log("PIPELINE COMPLETE")
    log("=" * 50)
    log(f"  Input shape                : {df_cleaned.shape}")
    log(f"  After correlation filter   : {df_reduced.shape}")
    log(f"  After VIF filter           : {df_multicol.shape}")
    log(f"  After outlier removal      : {df_final.shape}")
    log(f"  After scaling              : {df_scaled.shape}")
    log(f"  Outlier rows removed       : {outlier_summary['rows_removed']}")
    log(f"  Encoded columns (kept)     : {encoded_cols}")

    log("\nOutput files:")
    for f in [
        log_file,
        cleaned_file,
        encoded_intermediate,
        reduced_file,
        vif_file,
        dropped_vif_file,
        multicol_file,
        stats_file,
        scaled_file,
        final_file,
        removed_file,
        outlier_report,
    ]:
        log(f"  - {f}")

    log(f"  - {heatmap_fig}")
    log(f"  - {spearman_fig}")
    log(f"  - {boxplot_fig}")
    log(f"  - {qqplot_folder}/")

    log.close()
