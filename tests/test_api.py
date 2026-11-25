"""Tests for the FastAPI service (api/main.py) using FastAPI's TestClient.

TestClient sends real HTTP requests to the app in memory: no uvicorn, no
open port. Two things are swapped out with pytest's `monkeypatch` (changes
are undone automatically after each test):
  - MLflow: the app gets the model trained in conftest.py instead of
    downloading it, so the MLflow server isn't needed;
  - the database: each test logs to its own temporary SQLite file, so the
    real predictions.db is never touched and tests can't affect each other.
"""
from types import SimpleNamespace
from typing import get_args

import mlflow.sklearn
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import api.db
import api.main
from api.db import PredictionLog
from api.schemas import PatientFeatures

FAKE_MODEL_VERSION = 7  # MLflow returns an int; the API must turn it into "7"


class FakeMlflowClient:
    """Stands in for MlflowClient: answers the one call the API makes."""

    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version=FAKE_MODEL_VERSION)


@pytest.fixture
def test_engine(monkeypatch, tmp_path):
    """Point the API's database code at a temporary SQLite file.

    tmp_path is a fresh, empty folder that pytest creates for each test.
    """
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(api.db, "engine", engine)  # used by init_db()
    monkeypatch.setattr(api.main, "SessionLocal", sessionmaker(bind=engine))
    return engine


@pytest.fixture
def client(monkeypatch, test_engine, trained_pipeline):
    """A TestClient for the app with the model loaded."""
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: trained_pipeline)
    monkeypatch.setattr(api.main, "MlflowClient", FakeMlflowClient)

    # `with` runs the app's lifespan (startup and shutdown), exactly like uvicorn.
    with TestClient(api.main.app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch, test_engine):
    """A TestClient for the app when loading the model fails
    (e.g. the MLflow server is down)."""

    def failing_load(uri):
        raise ConnectionError("MLflow server unreachable")

    monkeypatch.setattr(mlflow.sklearn, "load_model", failing_load)

    with TestClient(api.main.app) as test_client:
        yield test_client


def logged_rows(engine) -> list[PredictionLog]:
    """Helper: every row in the prediction log."""
    with Session(engine) as session:
        return list(session.scalars(select(PredictionLog)))


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_ok_when_model_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == str(FAKE_MODEL_VERSION)  # int -> str


def test_health_503_when_model_missing(client_without_model):
    response = client_without_model.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["model_loaded"] is False
    assert "MLflow server unreachable" in body["error"]


# ---------------------------------------------------------------------------
# /predict: valid requests
# ---------------------------------------------------------------------------
def test_predict_returns_prediction(client, valid_patient):
    response = client.post("/predict", json=valid_patient)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Alive", "Dead"}
    assert 0 <= body["probability_dead"] <= 1
    assert body["model_version"] == str(FAKE_MODEL_VERSION)


def test_prediction_label_follows_threshold(client, valid_patient, high_risk_patient):
    for patient in (valid_patient, high_risk_patient):
        body = client.post("/predict", json=patient).json()

        expected = "Dead" if body["probability_dead"] >= api.main.THRESHOLD else "Alive"
        assert body["prediction"] == expected


def test_api_matches_the_model_directly(client, trained_pipeline, valid_patient):
    # No training/serving skew: the API must give exactly what the pipeline
    # gives for the same patient.
    import pandas as pd

    row = PatientFeatures(**valid_patient).model_dump(by_alias=True)
    expected = trained_pipeline.predict_proba(pd.DataFrame([row]))[0, 1]

    body = client.post("/predict", json=valid_patient).json()

    assert body["probability_dead"] == pytest.approx(expected, abs=1e-4)


def test_valid_prediction_is_logged(client, test_engine, valid_patient):
    body = client.post("/predict", json=valid_patient).json()

    rows = logged_rows(test_engine)
    assert len(rows) == 1
    row = rows[0]
    assert row.age == valid_patient["age"]
    assert row.t_stage == valid_patient["t_stage"]
    assert row.prediction == body["prediction"]
    assert row.model_version == str(FAKE_MODEL_VERSION)
    assert row.created_at is not None


# ---------------------------------------------------------------------------
# /predict: invalid requests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"t_stage": "T9"}, id="unknown-category"),
        pytest.param({"age": 5}, id="age-too-low"),
        pytest.param({"tumor_size": 0}, id="tumor-size-zero"),
        pytest.param({"age": "old"}, id="wrong-type"),
        pytest.param(
            {"regional_node_positive": 30, "regional_node_examined": 24},
            id="more-positive-than-examined",
        ),
        pytest.param({"tumour_size": 4}, id="unknown-field"),
    ],
)
def test_invalid_request_is_rejected_and_not_logged(
    client, test_engine, valid_patient, change
):
    response = client.post("/predict", json={**valid_patient, **change})

    assert response.status_code == 422  # invalid input, not a server error
    assert logged_rows(test_engine) == []  # never reached the model or the log


def test_missing_field_is_rejected(client, valid_patient):
    del valid_patient["t_stage"]

    response = client.post("/predict", json=valid_patient)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "t_stage"]


def test_predict_503_when_model_missing(client_without_model, test_engine, valid_patient):
    response = client_without_model.post("/predict", json=valid_patient)

    assert response.status_code == 503
    assert logged_rows(test_engine) == []


# ---------------------------------------------------------------------------
# Schema vs training data
# ---------------------------------------------------------------------------
def test_schema_categories_match_training_data(clean_df):
    # The API's allowed values must be exactly the values the model was
    # trained on: a value missing from the schema would be wrongly rejected;
    # an extra one would reach the model as an unknown category.
    for name, field in PatientFeatures.model_fields.items():
        allowed = get_args(field.annotation)  # the values inside Literal[...]
        if not allowed:
            continue  # numeric field
        column = field.serialization_alias  # e.g. t_stage -> "T Stage"

        assert set(allowed) == set(clean_df[column].unique()), name
