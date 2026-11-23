"""Exploratory data analysis of the SEER breast cancer dataset.

Run once to understand the data; findings drive the cleaning in src/data.py.
"""
import pandas as pd

DATA_PATH = "data/raw/Breast_Cancer.csv"

df = pd.read_csv(DATA_PATH)

print("=== Shape (rows, columns) ===")
print(df.shape)

print("\n=== Column names (repr shows hidden spaces) ===")
print([repr(c) for c in df.columns])

print("\n=== Data types ===")
print(df.dtypes)

print("\n=== Missing values per column ===")
print(df.isna().sum())

print("\n=== Duplicate rows ===")
print(df.duplicated().sum())

print("\n=== Target balance (Status) ===")
print(df["Status"].value_counts())
print(df["Status"].value_counts(normalize=True).round(3))

print("\n=== Numeric summary ===")
print(df.describe().round(1))

print("\n=== Categories in each text column (repr shows hidden spaces) ===")
for col in df.select_dtypes(exclude="number").columns:
    print(f"{col!r}: {[repr(v) for v in sorted(df[col].unique())]}")

print("\n=== Grade vs differentiate (are they the same information?) ===")
print(pd.crosstab(df["Grade"], df["differentiate"]))