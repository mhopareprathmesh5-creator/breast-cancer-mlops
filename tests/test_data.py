"""Tests for src/data.py: loading, cleaning and the feature/target split.

Two kinds of test:
  - unit tests on tiny hand-made DataFrames: each checks one cleaning rule;
  - data tests on the real CSV: check the cleaned dataset has the shape and
    content the model was designed for.
"""
import pandas as pd

from src.data import (
    DROP_COLUMNS,
    FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    TARGET,
    clean,
)


# ---------------------------------------------------------------------------
# Unit tests: one cleaning rule each, on tiny DataFrames
# ---------------------------------------------------------------------------
def test_clean_strips_spaces_from_column_names():
    df = pd.DataFrame({"T Stage ": ["T1"]})  # trailing space, as in the CSV

    assert list(clean(df).columns) == ["T Stage"]


def test_clean_strips_spaces_from_text_values():
    df = pd.DataFrame({"Marital Status": ["Single ", " Married"]})

    assert clean(df)["Marital Status"].tolist() == ["Single", "Married"]


def test_clean_leaves_numbers_unchanged():
    df = pd.DataFrame({"Age": [68, 50]})

    assert clean(df)["Age"].tolist() == [68, 50]


def test_clean_drops_exact_duplicate_rows():
    df = pd.DataFrame({"Age": [63, 63, 50], "Race": ["White", "White", "Black"]})

    assert len(clean(df)) == 2


def test_clean_finds_duplicates_after_stripping_spaces():
    # "Single" and "Single " are the same value once spaces are stripped.
    df = pd.DataFrame({"Marital Status": ["Single", "Single "]})

    assert len(clean(df)) == 1


def test_clean_does_not_modify_its_input():
    df = pd.DataFrame({"T Stage ": ["T1 "]})

    clean(df)

    assert list(df.columns) == ["T Stage "]  # the original is untouched
    assert df["T Stage "].tolist() == ["T1 "]


# ---------------------------------------------------------------------------
# Feature list consistency
# ---------------------------------------------------------------------------
def test_there_are_13_unique_features():
    assert len(FEATURES) == 13
    assert len(set(FEATURES)) == 13  # no feature listed twice


def test_feature_groups_add_up_to_all_features():
    groups = NUMERIC_FEATURES + list(ORDINAL_FEATURES) + NOMINAL_FEATURES

    assert sorted(groups) == sorted(FEATURES)


def test_leakage_redundant_and_target_columns_are_not_features():
    # Survival Months is only known after the outcome (leakage), Grade
    # duplicates differentiate, and the target must never be an input.
    for column in DROP_COLUMNS + [TARGET]:
        assert column not in FEATURES


# ---------------------------------------------------------------------------
# Data tests: the real CSV
# ---------------------------------------------------------------------------
def test_raw_data_has_expected_shape(raw_df):
    assert raw_df.shape == (4024, 16)


def test_cleaning_removes_exactly_one_duplicate(raw_df, clean_df):
    assert len(raw_df) - len(clean_df) == 1


def test_cleaned_data_has_no_missing_values(clean_df):
    assert clean_df.isna().sum().sum() == 0


def test_cleaned_column_names_have_no_surrounding_spaces(clean_df):
    for column in clean_df.columns:
        assert column == column.strip()


def test_features_are_exactly_the_model_inputs(features_target):
    X, _ = features_target

    assert list(X.columns) == FEATURES


def test_target_is_binary_with_about_15_percent_dead(features_target):
    _, y = features_target

    assert set(y.unique()) == {0, 1}
    assert 0.10 < y.mean() < 0.20  # 0.153 in the current data


def test_ordinal_values_are_all_in_the_declared_order(clean_df):
    # OrdinalEncoder fails on a value missing from its category list, so every
    # value in the data must appear in ORDINAL_FEATURES.
    for column, ordered_values in ORDINAL_FEATURES.items():
        assert set(clean_df[column]) <= set(ordered_values), column


def test_positive_nodes_never_exceed_examined_nodes(clean_df):
    # The API rejects requests that break this rule, so the training data
    # must follow it too.
    positive = clean_df["Reginol Node Positive"]
    examined = clean_df["Regional Node Examined"]

    assert (positive <= examined).all()
