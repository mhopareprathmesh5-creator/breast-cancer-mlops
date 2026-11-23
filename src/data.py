"""Load and clean the SEER breast cancer dataset.

Single source of truth for cleaning rules and feature lists, shared by
training (src/train.py), the API (api/) and the tests (tests/).
"""
from pathlib import Path

import pandas as pd

# Paths are built from this file's location, so scripts work from any folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Breast_Cancer.csv"

TARGET = "Status"
POSITIVE_CLASS = "Dead"  # encoded as 1: the class we care about catching

# Survival Months: only known after the outcome (data leakage).
# Grade: 1:1 duplicate of 'differentiate' (found in EDA).
DROP_COLUMNS = ["Survival Months", "Grade"]

NUMERIC_FEATURES = [
    "Age",
    "Tumor Size",
    "Regional Node Examined",
    "Reginol Node Positive",  # typo is in the original dataset
]

# Ordered categories, listed from least to most severe.
ORDINAL_FEATURES = {
    "T Stage": ["T1", "T2", "T3", "T4"],
    "N Stage": ["N1", "N2", "N3"],
    "6th Stage": ["IIA", "IIB", "IIIA", "IIIB", "IIIC"],
    "differentiate": [
        "Well differentiated",
        "Moderately differentiated",
        "Poorly differentiated",
        "Undifferentiated",
    ],
}

# Unordered categories: one-hot encoded, no fake order.
NOMINAL_FEATURES = [
    "Race",
    "Marital Status",
    "A Stage",
    "Estrogen Status",
    "Progesterone Status",
]

FEATURES = NUMERIC_FEATURES + list(ORDINAL_FEATURES) + NOMINAL_FEATURES


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as it is on disk."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the issues found in EDA. Returns a new DataFrame."""
    df = df.copy()
    df.columns = df.columns.str.strip()  # 'T Stage ' -> 'T Stage'
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()  # 'Single ' -> 'Single'
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (the 13 model features) and y (1 = Dead, 0 = Alive)."""
    X = df[FEATURES]
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)
    return X, y


def load_data(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean and split in one call."""
    return split_features_target(clean(load_raw(path)))


if __name__ == "__main__":
    X, y = load_data()
    print("X shape:", X.shape)
    print("Features:", list(X.columns))
    print("Dead rate:", round(y.mean(), 3))