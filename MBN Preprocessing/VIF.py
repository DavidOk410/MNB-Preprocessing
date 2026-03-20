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
    Iteratively remove VIF features with user input each round.

    After printing the VIF table the user is asked which column to drop.
    The loop stops automatically when all features are below `threshold`,
    or immediately if the user types 'done' or 'stop'.

    Parameters:
        df (pd.DataFrame)
        threshold (float)  : VIF cutoff — default 5
        protect_cols (list): columns never to drop (e.g. target / ID)
        exclude_cols (list): one-hot encoded columns — excluded from VIF
                             calculation and never dropped

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

        # Build numeric subset, excluding encoded & protected cols
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

        print(f"\n{'=' * 50}")
        print(f"  VIF Iteration {iteration}")
        print(f"{'=' * 50}")
        print(vif.to_string(index=False))

        max_vif = vif["VIF"].iloc[0]

        # Stop automatically if everything is already below threshold
        if max_vif <= threshold:
            print(f"\n  ✔ All features have VIF ≤ {threshold}. No more columns to drop.")
            break

        # List features above threshold (candidates to drop)
        above = vif[vif["VIF"] > threshold]["Feature"].tolist()
        print(f"\n  Features above threshold (VIF > {threshold}): {above}")
        print(f"  Suggested (highest VIF): '{vif['Feature'].iloc[0]}'  (VIF = {max_vif:.4f})")

        # Ask the user which column to drop
        while True:
            user_input = input(
                "\n  Enter column name to drop (or 'done'/'stop' to finish early): "
            ).strip()

            if user_input.lower() in ("done", "stop"):
                print("  Stopping VIF removal at user request.")
                print("\n" + "=" * 50)
                print(f"Total features dropped by VIF: {len(dropped_features)}")
                print(dropped_features)
                return X, dropped_features

            if user_input in vif["Feature"].values:
                if user_input in never_drop:
                    print(f"  ✗ '{user_input}' is protected and cannot be dropped. Choose another.")
                else:
                    break  # valid choice
            else:
                print(f"  ✗ '{user_input}' not found in the current feature list.")
                print(f"    Available: {vif['Feature'].tolist()}")

        print(f"\n  ➜ Dropping '{user_input}'")
        dropped_features.append(user_input)
        X = X.drop(columns=[user_input])

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
      3. Interactive iterative removal — user picks column to drop each round
      4. Save dropped-columns log
      5. Save final dataset (encoded columns preserved)

    Returns:
        df_final (pd.DataFrame)
        dropped_features (list)
    """
    print("\n--- Initial VIF values ---")
    initial_vif = compute_vif(df, exclude_cols=encoded_cols)
    print(initial_vif.to_string(index=False))

    initial_vif.to_excel(vif_output_path, index=False)
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