"""Tests for the model pipeline, including a quality gate.

The quality gate fails the test run (and later the Jenkins build) if the
model gets worse than an agreed minimum, so a change that hurts the model
can't reach production unnoticed.
"""
import numpy as np
import pandas as pd
import pytest

from api.schemas import PatientFeatures
from src.compare_models import build_pipeline, evaluate
from src.train import build_model

# Minimum acceptable scores on the held-out test set.
# Current model: recall 0.626, ROC-AUC 0.731. The minimums sit a little below,
# so the current model passes but a real drop in quality fails.
MIN_RECALL = 0.55  # main metric: share of Dead patients the model catches
MIN_ROC_AUC = 0.70  # ranking quality (0.5 = random guessing)


def risk_of(pipeline, patient: dict) -> float:
    """Helper: P(Dead) for one API-style patient dict."""
    row = PatientFeatures(**patient).model_dump(by_alias=True)
    return pipeline.predict_proba(pd.DataFrame([row]))[0, 1]


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
def test_model_meets_minimum_recall(trained_pipeline, data_split):
    _, X_test, _, y_test = data_split

    recall = evaluate(trained_pipeline, X_test, y_test)["recall"]

    assert recall >= MIN_RECALL, f"recall {recall:.3f} is below {MIN_RECALL}"


def test_model_meets_minimum_roc_auc(trained_pipeline, data_split):
    _, X_test, _, y_test = data_split

    roc_auc = evaluate(trained_pipeline, X_test, y_test)["roc_auc"]

    assert roc_auc >= MIN_ROC_AUC, f"ROC-AUC {roc_auc:.3f} is below {MIN_ROC_AUC}"


def test_model_flags_some_but_not_all_patients(trained_pipeline, data_split):
    # Guards against two useless extremes: predicting "Alive" for everyone
    # (0 recall) or "Dead" for everyone (100% recall but useless precision).
    _, X_test, _, _ = data_split

    flagged_share = trained_pipeline.predict(X_test).mean()

    assert 0.05 < flagged_share < 0.80


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------
def test_predicted_probabilities_are_valid(trained_pipeline, data_split):
    _, X_test, _, _ = data_split

    probabilities = trained_pipeline.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 2)  # [P(Alive), P(Dead)]
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_training_is_reproducible(data_split):
    # Same data + same seed must give the same model, otherwise MLflow runs
    # couldn't be compared or reproduced.
    X_train, X_test, y_train, _ = data_split

    first = build_pipeline(build_model()).fit(X_train, y_train)
    second = build_pipeline(build_model()).fit(X_train, y_train)

    np.testing.assert_allclose(
        first.predict_proba(X_test), second.predict_proba(X_test)
    )


def test_higher_risk_patient_gets_higher_risk(
    trained_pipeline, valid_patient, high_risk_patient
):
    # Medical common sense: the most severe stage, spread and receptor status
    # must score above a lower-risk patient.
    assert risk_of(trained_pipeline, high_risk_patient) > risk_of(
        trained_pipeline, valid_patient
    )


def test_unseen_category_does_not_crash(trained_pipeline, data_split):
    # OneHotEncoder(handle_unknown="ignore") turns an unknown value into all
    # zeros instead of raising an error.
    _, X_test, _, _ = data_split
    row = X_test.iloc[[0]].copy()
    row["Race"] = "Unknown"

    probability = trained_pipeline.predict_proba(row)[0, 1]

    assert 0 <= probability <= 1


@pytest.mark.parametrize("missing_column", ["T Stage", "Tumor Size"])
def test_missing_feature_column_raises_error(
    trained_pipeline, data_split, missing_column
):
    # Predicting from incomplete data must fail loudly, never silently.
    # (The API's schema already blocks this before it reaches the model.)
    _, X_test, _, _ = data_split
    incomplete = X_test.drop(columns=missing_column).iloc[[0]]

    with pytest.raises(ValueError):
        trained_pipeline.predict(incomplete)
