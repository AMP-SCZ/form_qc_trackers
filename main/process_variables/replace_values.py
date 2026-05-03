import pandas as pd


def fill_from_source(df_target, df_source, key_cols):
    """
    For every row in df_target whose `key_cols` values match a row in df_source,
    overwrite the other shared columns in df_target with the values from df_source.

    Parameters
    ----------
    df_target : pd.DataFrame
        The dataframe that will be updated (a copy is returned; the original is
        not mutated).
    df_source : pd.DataFrame
        The dataframe whose values take precedence on matched rows.
    key_cols : list[str] | tuple[str, str]
        The two (or more) column names used to identify a matching row.

    Returns
    -------
    pd.DataFrame
        A copy of df_target with matched rows updated.
    """
    key_cols = list(key_cols)

    missing_t = [c for c in key_cols if c not in df_target.columns]
    missing_s = [c for c in key_cols if c not in df_source.columns]
    if missing_t or missing_s:
        raise KeyError(
            f"Key columns missing — target: {missing_t}, source: {missing_s}"
        )

    # Columns to overwrite: shared between the two frames, excluding the keys.
    shared = [c for c in df_target.columns if c in df_source.columns and c not in key_cols]
    if not shared:
        return df_target.copy()

    # Keep only the first match per key combination on the source side so the
    # merge stays 1:1 from target's perspective.
    src = df_source[key_cols + shared].drop_duplicates(subset=key_cols, keep="first")

    merged = df_target.merge(
        src,
        on=key_cols,
        how="left",
        suffixes=("", "__src"),
    )

    result = df_target.copy()
    for col in shared:
        src_col = merged[f"{col}__src"]
        # Only overwrite where the source actually had a matching row.
        mask = src_col.notna()
        result.loc[mask, col] = src_col[mask].values

    result.to_csv('AMPSCZ-combined-redcap_month_2_PRESCIENT-day1to1.csv',
    index = False)

    return result


if __name__ == "__main__":
    # Quick demo / sanity check.
    df_target = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "batch": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
            "label": ["x", "y", "z", "w"],
        }
    )
    df_source = pd.DataFrame(
        {
            "id": [1, 3, 5],
            "batch": ["A", "B", "C"],
            "value": [100, 300, 500],
            "label": ["X", "Z", "Q"],
        }
    )

    print("target:")
    print(df_target)
    print("\nsource:")
    print(df_source)
    print("\nresult (matched on id+batch, filled from source):")
