"""Smoke test: is the deployed API really up and answering?

Run after deployment (the last pipeline stage). Unlike the pytest suite, this
talks to the REAL running container over HTTP, with the REAL model loaded
from MLflow. Any failure exits with an error, which fails the Jenkins build.

Usage:  python scripts/smoke_check.py [base_url]     (default http://localhost:8000)
"""
import json
import sys
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

PATIENT = {
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


def request(path: str, body: dict | None = None) -> dict:
    """GET (no body) or POST (JSON body); return the JSON response.

    Raises an error for any non-2xx status, which fails the smoke test.
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE_URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    # 1. Health: the model must be loaded.
    health = request("/health")
    assert health["model_loaded"] is True, f"model not loaded: {health}"
    print(f"Health OK: model version {health['model_version']}")

    # 2. A real prediction with a known patient.
    result = request("/predict", PATIENT)
    assert result["prediction"] in {"Alive", "Dead"}, result
    assert 0 <= result["probability_dead"] <= 1, result
    print(f"Prediction OK: {result}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
