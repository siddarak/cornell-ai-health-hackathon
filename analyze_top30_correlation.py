#!/usr/bin/env python3
"""
Analyze ed2021_2022_top30_correlation.csv: structure, composition, and relationship to ADMIT.
Run: python analyze_top30_correlation.py
"""

import pandas as pd
import numpy as np

INPUT_FILE = "ed2021_2022_top30_correlation.csv"


def load_data():
    """Load CSV; coerce byte strings and sentinel values for analysis."""
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    # Coerce object columns to numeric where possible (e.g. ARRTIME mixed int/string)
    for c in df.columns:
        if df[c].dtype == object and df[c].notna().any():
            try:
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notna().sum() > df[c].notna().sum() * 0.5:
                    df[c] = converted
            except Exception:
                pass
    return df


def main():
    print("=" * 60)
    print("ED 2021–2022 top-30 correlation dataset — summary analysis")
    print("=" * 60)

    df = load_data()
    n, p = df.shape
    print(f"\n1. SHAPE & MEMORY")
    print(f"   Rows: {n:,}  |  Columns: {p}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\n2. COLUMN LIST (order as in file)")
    for i, c in enumerate(df.columns, 1):
        print(f"   {i:2}. {c}")

    print(f"\n3. DTYPES")
    print(df.dtypes.to_string())

    print(f"\n4. SENTINEL / MISSING CODES (NHAMCS: -9=blank, -8=unknown, -7=N/A)")
    sentinels = [-9, -8, -7]
    for code in sentinels:
        counts = (df == code).sum()
        cols_with = counts[counts > 0]
        if len(cols_with) > 0:
            print(f"   {code}: {len(cols_with)} columns contain this value")
            for col in cols_with.index[:10]:
                print(f"        {col}: {cols_with[col]:,}")
            if len(cols_with) > 10:
                print(f"        ... and {len(cols_with) - 10} more columns")

    print(f"\n5. NULL / NaN COUNTS (after sentinels, or native NaN)")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if len(nulls) == 0:
        print("   No native NaNs in the file.")
    else:
        for col in nulls.index:
            print(f"   {col}: {nulls[col]:,} ({100 * nulls[col] / n:.1f}%)")

    # Target column
    if "ADMIT" in df.columns:
        print(f"\n6. TARGET: ADMIT (value counts)")
        vc = df["ADMIT"].value_counts().sort_index()
        for val, count in vc.items():
            print(f"   {val}: {count:,} ({100 * count / n:.1f}%)")
        print(f"   Dtype: {df['ADMIT'].dtype}")

    print(f"\n7. NUMERIC SUMMARY (describe)")
    num = df.select_dtypes(include=[np.number])
    if len(num.columns) > 0:
        desc = num.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        pd.set_option("display.max_columns", 8)
        pd.set_option("display.width", 120)
        pd.set_option("display.max_rows", 40)
        print(desc.to_string())

    print(f"\n8. UNIQUE VALUES PER COLUMN (top 15 by uniqueness)")
    uniques = df.nunique()
    uniques = uniques.sort_values(ascending=False).head(15)
    for col in uniques.index:
        print(f"   {col}: {uniques[col]:,} unique")

    # Correlation with ADMIT if numeric
    if "ADMIT" in df.columns and pd.api.types.is_numeric_dtype(df["ADMIT"]):
        print(f"\n9. CORRELATION WITH ADMIT (numeric columns only)")
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c != "ADMIT"]
        if num_cols:
            corr = df[num_cols].corrwith(df["ADMIT"]).abs().sort_values(ascending=False)
            for col in corr.head(15).index:
                r = df[col].corr(df["ADMIT"])
                print(f"   {col}: {r:.4f}")
            print("   (Showing top 15 by absolute correlation.)")

    print(f"\n10. KEY CATEGORICAL-LIKE COLUMNS (value counts)")
    for col in ["IMMEDR", "ARREMS", "RACERETH", "RESIDNCE"]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts().sort_index().head(12)
        print(f"\n   {col}:")
        for val, count in vc.items():
            print(f"      {val}: {count:,}")

    print(f"\n11. HOW THE DATA IS MADE UP (composition)")
    print(f"   - Rows: ED visits (2021 + 2022 combined sample).")
    print(f"   - Columns: 31 features chosen for high correlation with admission-related outcome (ADMIT).")
    print(f"   - Contains demographics (AGE, AGER, RACERETH), triage/arrival (IMMEDR, ARREMS, ARRTIME),")
    print(f"     comorbidities (COPD, HTN, CHF, CKD, CAD, etc.), payment (PAYMCARE, PAYMCAID),")
    print(f"     reason-for-visit (RFV4, RFV5, RFV53D), and vitals/other (BPSYS, RESPR, BAC).")
    print(f"   - Sentinel values -9/-8/-7 indicate missing/unknown/N/A; replace with NaN before modeling.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
