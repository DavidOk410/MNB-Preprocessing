import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer


def load_data(file_path):
    """Load raw Excel dataset."""
    df = pd.read_excel(file_path)
    print(f"Initial shape: {df.shape}")
    return df


def report_missing(df):
    """Print a missing values report and return missing percentages."""
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100

    missing_report = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing %': missing_percent
    })

    print("\nMissing values per column:\n")
    print(missing_report)

    return missing_percent


def plot_missingness_heatmap(df, save_path="missingness_heatmap.png"):
    """Generate and save the missingness heatmap."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

    plt.title("Missing Data Heatmap", fontsize=12)
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nMissingness heatmap saved as '{save_path}'")


def drop_high_missing_cols(df, missing_percent, threshold=50):
    """Drop columns with more than `threshold`% missing values."""
    cols_to_drop = missing_percent[missing_percent > threshold].index
    df_cleaned = df.drop(columns=cols_to_drop)

    print(f"\nDropped columns (>{threshold}% missing):")
    print(list(cols_to_drop))
    print("Shape after dropping:", df_cleaned.shape)

    return df_cleaned


def encode_categorical(df, protect_col="Paper"):
    """
    One-hot encode all categorical columns except `protect_col`.
    Also convert boolean columns (True/False) to 1/0.

    Returns:
        df_encoded (pd.DataFrame): dataframe with encoded columns appended
        encoded_cols (list): names of the newly created one-hot columns
    """

    # --- Convert boolean columns to 0/1 ---
    # Case 1: actual bool dtype
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()

    # Case 2: object columns where every non-null cell is True/False (any form)
    object_bool_cols = [
        col for col in df.select_dtypes(include='object').columns
        if df[col].dropna().isin([True, False, 'True', 'False']).all()
    ]

    all_bool_cols = list(set(bool_cols + object_bool_cols))

    for col in all_bool_cols:
        df[col] = df[col].apply(lambda x: 1 if x in (True, 'True') else (0 if x in (False, 'False') else x))

    print("\nBoolean columns converted to 0/1:")
    print(df[all_bool_cols])

    # --- Identify categorical columns ---
    categorical_cols = (
        df.drop(columns=[protect_col], errors='ignore')
          .select_dtypes(include=['object', 'category'])
          .columns.tolist()
    )

    print("\nCategorical columns to encode:")
    print(categorical_cols)

    # --- One-hot encoding ---
    cols_before = set(df.columns)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    encoded_cols = [c for c in df_encoded.columns if c not in cols_before]

    # ✅ get_dummies produces bool dtype — convert to int immediately
    df_encoded[encoded_cols] = df_encoded[encoded_cols].astype(int)

    print(f"One-hot encoded columns created ({len(encoded_cols)}):")
    print(encoded_cols)
    print("Shape after encoding:", df_encoded.shape)

    return df_encoded, encoded_cols

def knn_impute(df, n_neighbors=5):
    """Impute missing values in numeric columns using KNN."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    print("\nMissing values AFTER imputation:")
    print(df.isnull().sum())
    print("\nShape after imputation:", df.shape)

    return df


def run_missingness_pipeline(file_path, output_path="Cleaned_dataset Task 2.xlsx"):
    """
    Full missingness pipeline:
      1. Load data
      2. Report & visualise missing values
      3. Drop high-missing columns (>50%)
      4. One-hot encode categorical columns (except 'Paper')
      5. KNN imputation on numeric columns
      6. Save cleaned dataset

    Returns:
        df_cleaned   (pd.DataFrame) : fully cleaned & encoded dataframe
        encoded_cols (list)         : names of one-hot encoded columns
                                      (passed downstream so other steps can ignore them)
    """
    df = load_data(file_path)
    missing_percent = report_missing(df)
    plot_missingness_heatmap(df)
    df_cleaned = drop_high_missing_cols(df, missing_percent)
    df_cleaned, encoded_cols = encode_categorical(df_cleaned)
    df_cleaned = knn_impute(df_cleaned)

    df_cleaned.to_excel(output_path, index=False)
    print(f"\nDataset saved as '{output_path}'")

    return df_cleaned, encoded_cols