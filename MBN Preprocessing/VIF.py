import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df, exclude_cols=None):
    """
    Compute VIF for numeric columns, excluding `exclude_cols`.

    Returns:
        vif_data (pd.DataFrame): Feature and VIF columns, sorted descending
    """
    if exclude_cols is None:
        exclude_cols = []

    X = (
        df.select_dtypes(include=[np.number])
          .drop(columns=exclude_cols, errors='ignore')
          .copy()
    )
    X['intercept'] = 1

    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    vif_data = (
        vif_data[vif_data["Feature"] != "intercept"]
               .sort_values("VIF", ascending=False)
               .reset_index(drop=True)
    )

    return vif_data


def remove_high_vif(df, threshold=5, protect_cols=None, exclude_cols=None):
    """
    Iteratively show VIF table and ask the user what to do each round.

    - High VIF features are flagged but the user decides whether to drop anything.
    - Input options each round:
        • A number (1, 2, 3 …) → drop the feature at that row in the table
        • A column name        → drop that specific column
        • 'keep'               → proceed to the next iteration without dropping
        • 'done' / 'stop'      → exit immediately, keep current state

    The loop ends automatically when all features are below `threshold` OR
    the user chooses to stop.

    Parameters:
        df (pd.DataFrame)
        threshold (float)  : VIF warning threshold — default 5
        protect_cols (list): columns never to drop (e.g. target / ID)
        exclude_cols (list): one-hot encoded columns — excluded from VIF,
                             never dropped

    Returns:
        df_final (pd.DataFrame)
        dropped_features (list)
    """
    if protect_cols is None:
        protect_cols = []
    if exclude_cols is None:
        exclude_cols = []

    never_drop = set(protect_cols) | set(exclude_cols)

    X = df.copy()
    dropped_features = []
    iteration = 0

    while True:
        iteration += 1

        # Build numeric subset excluding protected/encoded cols
        numeric_X = (
            X.select_dtypes(include=[np.number])
             .drop(columns=list(never_drop), errors='ignore')
             .copy()
        )
        numeric_X['intercept'] = 1

        vif = pd.DataFrame({
            "Feature": numeric_X.columns,
            "VIF": [
                variance_inflation_factor(numeric_X.values, i)
                for i in range(numeric_X.shape[1])
            ]
        })
        vif = (
            vif[vif["Feature"] != "intercept"]
               .sort_values("VIF", ascending=False)
               .reset_index(drop=True)
        )

        # Add 1-based index column for easy reference
        vif.index = vif.index + 1
        vif.index.name = "#"

        print(f"\n{'=' * 50}")
        print(f"  VIF Iteration {iteration}")
        print(f"{'=' * 50}")
        print(vif.to_string())

        max_vif = vif["VIF"].iloc[0]
        above = vif[vif["VIF"] > threshold]

        if not above.empty:
            print(f"\n  ⚠ {len(above)} feature(s) have VIF > {threshold}:")
            for idx, row in above.iterrows():
                print(f"    [{idx}] {row['Feature']}  (VIF = {row['VIF']:.4f})")
        else:
            print(f"\n  ✔ All features have VIF ≤ {threshold}.")

        print("\n  Options:")
        print("    • Enter a number (e.g. '1') to drop that row's feature")
        print("    • Enter a column name to drop it directly")
        print("    • 'done'/'stop' → exit VIF removal now")

        # User input loop
        while True:
            user_input = input("\n  Your choice: ").strip()

            # Exit early
            if user_input.lower() in ("done", "stop"):
                print("  Exiting VIF removal.")
                print("\n" + "=" * 50)
                print(f"Total features dropped by VIF: {len(dropped_features)}")
                print(dropped_features)
                return X, dropped_features

            # Numeric shortcut — resolve to column name
            if user_input.isdigit():
                row_num = int(user_input)
                if row_num in vif.index:
                    col_to_drop = vif.loc[row_num, "Feature"]
                    if col_to_drop in never_drop:
                        print(f"  ✗ '{col_to_drop}' is protected and cannot be dropped.")
                        continue
                    print(f"\n  ➜ Dropping [{row_num}] '{col_to_drop}'  (VIF = {vif.loc[row_num, 'VIF']:.4f})")
                    dropped_features.append(col_to_drop)
                    X = X.drop(columns=[col_to_drop])
                    break
                else:
                    print(f"  ✗ No row #{row_num} in the table. Valid range: 1–{len(vif)}.")
                    continue

            # Column name input
            if user_input in vif["Feature"].values:
                if user_input in never_drop:
                    print(f"  ✗ '{user_input}' is protected and cannot be dropped.")
                    continue
                row_num = vif[vif["Feature"] == user_input].index[0]
                print(f"\n  ➜ Dropping '{user_input}'  (VIF = {vif.loc[row_num, 'VIF']:.4f})")
                dropped_features.append(user_input)
                X = X.drop(columns=[user_input])
                break
            else:
                print(f"  ✗ '{user_input}' not recognised.")
                print(f"    Enter a number (1–{len(vif)}), a column name, 'keep', 'done', or 'stop'.")

    print("\n" + "=" * 50)
    print(f"Total features dropped by VIF: {len(dropped_features)}")
    print(dropped_features)

    return X, dropped_features


def run_vif_pipeline(
    df,
    threshold=5,
    protect_cols=None,
    encoded_cols=None,
    vif_output_path="VIF_results.xlsx",
    dropped_output_path="Dropped_VIF_columns.xlsx",
    final_output_path="Final_dataset_Multicol.xlsx"
):
    """
    Full VIF pipeline:
      1. Compute & print initial VIF (excluding encoded columns)
      2. Save initial VIF report
      3. Interactive iterative removal — user controls every drop
      4. Save dropped-columns log
      5. Save final dataset (encoded columns preserved)

    Returns:
        df_final (pd.DataFrame)
        dropped_features (list)
    """
    print("\n--- Initial VIF values ---")
    initial_vif = compute_vif(df, exclude_cols=encoded_cols)
    # Show with 1-based index
    initial_vif.index = initial_vif.index + 1
    initial_vif.index.name = "#"
    print(initial_vif.to_string())

    initial_vif.to_excel(vif_output_path)
    print(f"\nInitial VIF results saved as '{vif_output_path}'")

    df_final, dropped_features = remove_high_vif(
        df,
        threshold=threshold,
        protect_cols=protect_cols,
        exclude_cols=encoded_cols
    )

    pd.DataFrame({"Dropped Columns": dropped_features}).to_excel(dropped_output_path, index=False)
    print(f"Dropped VIF columns log saved as '{dropped_output_path}'")

    df_final.to_excel(final_output_path, index=False)
    print(f"Final dataset saved as '{final_output_path}'")
    print("Final shape after VIF filtering:", df_final.shape)

    return df_final, dropped_features