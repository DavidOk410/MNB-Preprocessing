"""
outliers.py — Outlier detection, removal, and detailed reporting.

Changes from original:
  - run_outlier_pipeline() now accepts a `report_path` argument.
  - Detailed per-row outlier information is written to a human-readable .txt
    report file instead of being printed to the terminal.
  - Terminal output is kept minimal: only confirmation messages are printed.
  - qq_plots() and remove_outliers_zscore() are unchanged in logic.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import zscore
from datetime import datetime


# ---------------------------------------------------------------------------
# Q-Q plots
# ---------------------------------------------------------------------------

def qq_plots(df, exclude_cols=None, save=True, folder="qqplots", threshold=3):
    """
    Generate Q-Q plots for original numeric columns (encoded columns skipped).

    Dots whose |Z-score| > threshold are highlighted in red; normal dots in blue.

    Parameters
    ----------
    df           : pd.DataFrame
    exclude_cols : list  — one-hot encoded column names to skip
    save         : bool
    folder       : str   — output directory for plot files
    threshold    : float — Z-score cutoff used to colour outlier dots red
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

        (osm, osr), (slope, intercept, _) = stats.probplot(series, dist="norm")

        col_zscore = np.abs(zscore(series, nan_policy='omit'))
        sorted_zscores = col_zscore[np.argsort(series)]
        is_outlier = sorted_zscores > threshold

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(osm[~is_outlier], osr[~is_outlier],
                   color="steelblue", s=20, zorder=3, label="Normal")

        if is_outlier.any():
            ax.scatter(osm[is_outlier], osr[is_outlier],
                       color="red", s=30, zorder=4,
                       label=f"Outlier (|Z| > {threshold})")

        x_line = np.array([osm.min(), osm.max()])
        ax.plot(x_line, slope * x_line + intercept,
                color="black", linewidth=1, zorder=2)

        n_outliers = int(is_outlier.sum())
        ax.set_title(f"Q-Q Plot: {col}\n({n_outliers} outlier(s) highlighted)",
                     fontsize=9)
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

    print(f"  Q-Q plots generated for {len(plot_cols)} variable(s)"
          + (f" — saved in '{folder}/'" if save else ""))


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers_zscore(df, threshold=3, exclude_cols=None, paper_col="Paper"):
    """
    Remove rows where any *original* numeric column has |Z-score| > threshold.

    Parameters
    ----------
    df           : pd.DataFrame
    threshold    : float — Z-score cutoff (default 3)
    exclude_cols : list  — encoded column names excluded from scoring
    paper_col    : str   — column name used as the row identifier in reports
                           (e.g. 'Paper'); falls back to the integer row index
                           if the column is absent

    Returns
    -------
    df_clean     : pd.DataFrame — rows that passed the filter
    removed_rows : pd.DataFrame — rows that were removed
    summary      : dict         — row counts and per-row outlier detail
    """
    if exclude_cols is None:
        exclude_cols = []

    score_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    z_scores_arr = np.abs(zscore(df[score_cols], nan_policy='omit'))
    z_df = pd.DataFrame(z_scores_arr, columns=score_cols, index=df.index)

    mask = (z_df < threshold).all(axis=1)

    rows_before  = df.shape[0]
    df_clean     = df[mask].copy()
    removed_rows = df[~mask].copy()
    rows_after   = df_clean.shape[0]

    # --- Build per-row detail for the report ---
    outlier_details = []
    for idx in removed_rows.index:
        # Resolve human-readable paper identifier
        if paper_col in df.columns:
            paper_id = df.loc[idx, paper_col]
        else:
            paper_id = f"Row {idx}"

        offending = {
            col: {
                "value":   df.loc[idx, col],
                "z_score": float(z_df.loc[idx, col])
            }
            for col in score_cols
            if z_df.loc[idx, col] > threshold
        }
        outlier_details.append({
            "row_index":         idx,
            "paper":             paper_id,
            "offending_columns": offending,
        })

    summary = {
        "rows_before":     rows_before,
        "rows_after":      rows_after,
        "rows_removed":    rows_before - rows_after,
        "score_cols":      score_cols,
        "threshold":       threshold,
        "outlier_details": outlier_details,
    }

    return df_clean, removed_rows, summary


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_outlier_report(summary, report_path):
    """
    Write a compact, human-readable text report of outlier removal results.

    Each removed row occupies exactly two lines:
      Line 1 — paper identifier, row index, number of flagged columns
      Line 2 — every offending column with its value and Z-score

    Example
    -------
      [01]  Paper: Smith et al. (2021)   |   Row 42   |   2 column(s) flagged
            Age: 87.30  (Z=3.41)  ·  BMI: 51.20  (Z=3.88)

    Parameters
    ----------
    summary     : dict — as returned by remove_outliers_zscore()
    report_path : str  — destination .txt file path
    """
    threshold  = summary["threshold"]
    details    = summary["outlier_details"]
    score_cols = summary["score_cols"]

    lines = []
    lines.append("=" * 80)
    lines.append("  OUTLIER REMOVAL REPORT")
    lines.append(f"  Generated  :  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Method     :  Z-score  |  Threshold: |Z| > {threshold}  →  row removed")
    lines.append(f"  Scored on  :  {len(score_cols)} column(s)")
    lines.append(f"  Removed    :  {summary['rows_removed']} of {summary['rows_before']} rows")
    lines.append("=" * 80)

    if not details:
        lines.append("")
        lines.append("  No outlier rows were detected.")
    else:
        lines.append("")

        for i, entry in enumerate(details, start=1):
            paper    = entry["paper"]
            row_idx  = entry["row_index"]
            offending = entry["offending_columns"]

            # --- Line 1: identifier + count ---
            lines.append(
                f"  [{i:02d}]  Paper: {paper}"
                f"   |   Row {row_idx}"
                f"   |   {len(offending)} column(s) flagged"
            )

            # --- Line 2: all offending columns on one line ---
            col_parts = [
                f"{col}: {info['value']:.4g}  (Z={info['z_score']:.2f})"
                for col, info in offending.items()
            ]
            lines.append("        " + "  ·  ".join(col_parts))
            lines.append("")

    lines.append("=" * 80)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"  Outlier report saved  : {report_path}")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_outlier_pipeline(
    df,
    encoded_cols=None,
    threshold=3,
    save_plots=True,
    output_path="05_final_no_outliers.xlsx",
    removed_output_path="05_removed_outliers.xlsx",
    report_path="05_outliers_report.txt",
    paper_col="Paper",
    qqplot_folder="qqplots",
):
    """
    Full outlier pipeline:
      1. Generate Q-Q plots (original numeric columns only)
      2. Remove outliers via Z-score filtering
      3. Save cleaned dataset to Excel
      4. Save removed outlier rows to a separate Excel file
      5. Write a compact two-line-per-row text report

    Terminal output is intentionally minimal — all detail goes to the report file.

    Parameters
    ----------
    df                  : pd.DataFrame — dataset entering this stage
    encoded_cols        : list  — one-hot column names to skip
    threshold           : float — Z-score cutoff (default 3)
    save_plots          : bool  — whether to save Q-Q plot images
    output_path         : str   — Excel path for the cleaned dataset
    removed_output_path : str   — Excel path for removed rows
    report_path         : str   — .txt path for the outlier report
    paper_col           : str   — column used as the paper/row identifier
                                  in the report (default 'Paper')

    Returns
    -------
    df_clean     : pd.DataFrame
    removed_rows : pd.DataFrame
    summary      : dict
    """
    print(f"  Generating Q-Q plots ...")
    qq_plots(df, exclude_cols=encoded_cols, save=save_plots,
             folder=qqplot_folder, threshold=threshold)

    print(f"  Running Z-score outlier removal (threshold = {threshold}) ...")
    df_clean, removed_rows, summary = remove_outliers_zscore(
        df, threshold=threshold, exclude_cols=encoded_cols, paper_col=paper_col
    )

    # Save datasets
    df_clean.to_excel(output_path, index=False)
    removed_rows.to_excel(removed_output_path, index=False)
    print(f"  Cleaned dataset saved : {output_path}")
    print(f"  Removed rows saved    : {removed_output_path}")

    # Write compact report to file (not to terminal)
    write_outlier_report(summary, report_path)

    print(f"  Rows removed          : {summary['rows_removed']} "
          f"(see report for details)")

    return df_clean, removed_rows, summary