"""Shared pytest fixtures.

pytest loads this file automatically before the tests. A fixture is a
function that prepares something a test needs (data, a trained model, ...);
a test asks for it simply by naming it as an argument.

scope="session" means the fixture runs ONCE for the whole test run and the
result is shared, so the CSV is read once and the model is trained once.

Run all tests from the project root with:  python -m pytest
"""
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.compare_models import RANDOM_STATE, TEST_SIZE, build_pipeline
from src.data import RAW_DATA_PATH, clean, load_raw, split_features_target
from src.train import build_model


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The CSV exactly as on disk.

    The dataset is not in Git, so fail with a clear instruction rather than
    silently skipping: a CI run that skips the data and model checks would
    show "passed" without having tested anything.
    """
    if not RAW_DATA_PATH.exists():
        pytest.fail(
            f"Dataset not found at {RAW_DATA_PATH}. Download Breast_Cancer.csv "
            "from Kaggle (reihanenamdari/breast-cancer) into data/raw/."
        )
    return load_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """The cleaned dataset (spaces stripped, duplicate dropped)."""
    return clean(raw_df)


@pytest.fixture(scope="session")
def features_target(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X (13 features) and y (1 = Dead, 0 = Alive)."""
    return split_features_target(clean_df)


@pytest.fixture(scope="session")
def data_split(features_target):
    """The same stratified 80/20 split that training uses:
    (X_train, X_test, y_train, y_test)."""
    X, y = features_target
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


@pytest.fixture(scope="session")
def trained_pipeline(data_split):
    """The production pipeline (preprocessing + Logistic Regression) trained
    on the training split. Built with the same functions as src/train.py but
    without MLflow, so the tests don't need the MLflow server."""
    X_train, _, y_train, _ = data_split
    return build_pipeline(build_model()).fit(X_train, y_train)


@pytest.fixture
def valid_patient() -> dict:
    """A valid /predict request body (a lower-risk patient).

    No scope -> a fresh copy for every test, so a test that changes it can't
    affect another test.
    """
    return {
        "age": 68,
        "race": "White",
        "marital_status": "Married",
        "t_stage": "T1",
        "n_stage": "N1",
        "sixth_stage": "IIA",
        "differentiate": "Poorly differentiated",
        "a_stage": "Regional",
        "tumor_size": 4,
        "estrogen_status": "Positive",
        "progesterone_status": "Positive",
        "regional_node_examined": 24,
        "regional_node_positive": 1,
    }


@pytest.fixture
def high_risk_patient(valid_patient: dict) -> dict:
    """The same patient with every risk factor at its most severe."""
    return {
        **valid_patient,
        "t_stage": "T4",
        "n_stage": "N3",
        "sixth_stage": "IIIC",
        "differentiate": "Undifferentiated",
        "a_stage": "Distant",
        "tumor_size": 90,
        "estrogen_status": "Negative",
        "progesterone_status": "Negative",
        "regional_node_positive": 20,
    }
