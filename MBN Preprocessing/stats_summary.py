import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def compute_statistics(df, exclude_cols=None):
    """
    Compute statistical characteristics for all remaining numeric features.

    Reports: Mean, Median, Q1, Q3, Min, Max, Variance, SD, Skewness, Kurtosis

    Parameters:
        df (pd.DataFrame)
        exclude_cols (list): columns to skip (e.g. encoded, ID columns)

    Returns:
        summary (pd.DataFrame): features as rows, statistics as columns
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    records = []
    for col in numeric_cols:
        s = df[col].dropna()
        records.append({
            "Feature"  : col,
            "Mean"     : s.mean(),
            "Median"   : s.median(),
            "Q1"       : s.quantile(0.25),
            "Q3"       : s.quantile(0.75),
            "Min"      : s.min(),
            "Max"      : s.max(),
            "Variance" : s.var(),
            "SD"       : s.std(),
            "Skewness" : skew(s),
            "Kurtosis" : kurtosis(s),   # excess kurtosis (normal = 0)
        })

    summary = pd.DataFrame(records).set_index("Feature")
    return summary


def run_stats_pipeline(
    df,
    exclude_cols=None,
    output_path="Statistical_summary.xlsx"
):
    """
    Compute and save statistical summary for all numeric features.

    Parameters:
        df (pd.DataFrame): final cleaned dataset
        exclude_cols (list): columns to exclude (e.g. one-hot encoded cols)
        output_path (str): path to save the Excel report

    Returns:
        summary (pd.DataFrame)
    """
    summary = compute_statistics(df, exclude_cols=exclude_cols)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 60)
    print("  Statistical Feature Characteristics")
    print("=" * 60)
    print(summary.to_string())

    summary.to_excel(output_path)
    print(f"\nStatistical summary saved as '{output_path}'")

    return summary
