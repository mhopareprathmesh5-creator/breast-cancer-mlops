# Breast Cancer Survival Classification — MLOps Pipeline

End-to-end MLOps project: predict patient survival status (Alive / Dead) from
clinical features in the SEER Breast Cancer dataset, with experiment tracking,
an API, automated tests, containers, CI/CD and drift monitoring.

## Tech stack
| Area | Tool |
|---|---|
| Modelling | Python 3.11, scikit-learn |
| Experiment tracking | MLflow |
| Serving | FastAPI |
| Testing | pytest |
| Containers | Docker, docker-compose |
| CI/CD | Jenkins |
| Monitoring | Evidently |

## Dataset
SEER Breast Cancer data (Kaggle: `reihanenamdari/breast-cancer`), ~4,024 patients,
16 columns. ~15% of patients are in the "Dead" class, so the main metric is
**recall on the Dead class** rather than accuracy.

The data is **not committed** to this repo. Download `Breast_Cancer.csv` from
Kaggle and place it in `data/raw/`.

## Setup (Windows / PowerShell)
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Project structure
```
src/         data cleaning, model comparison, training (MLflow)
api/         FastAPI service (schemas, prediction log, endpoints)
tests/       pytest tests (data, model quality gate, API)
monitoring/  Evidently drift reports
data/raw/    dataset (git-ignored)
Dockerfile, docker-compose.yml   container image and the 3-service stack
```

## Train
Start the MLflow tracking server in a separate terminal (UI at http://127.0.0.1:5000):
```powershell
mlflow server
```

Train the production model (Logistic Regression), log it to MLflow and register it
as `breast-cancer-survival@champion`:
```powershell
python -m src.train
```

### Model selection
`python -m src.compare_models` compares three candidates (held-out 20% test set,
Dead = positive class). Logistic Regression won and is the production model:

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.626 | 0.280 | 0.387 | 0.731 |
| Random Forest | 0.537 | 0.297 | 0.383 | 0.713 |
| Gradient Boosting | 0.415 | 0.282 | 0.336 | 0.689 |

5-fold cross-validated recall confirms it: LR 0.631 ± 0.030 (RF 0.517, GB 0.391).
Rerun the comparison when the data changes or drift is detected.

## Run the API
With `mlflow server` running (the API loads `breast-cancer-survival@champion` from it at startup):
```powershell
python -m uvicorn api.main:app --reload
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness: `200` when the model is loaded, `503` otherwise |
| `POST /predict` | Risk prediction for one patient |
| `GET /docs` | Interactive Swagger UI: http://127.0.0.1:8000/docs |

Example request to `/predict`:
```json
{
  "age": 68, "race": "White", "marital_status": "Married",
  "t_stage": "T1", "n_stage": "N1", "sixth_stage": "IIA",
  "differentiate": "Poorly differentiated", "a_stage": "Regional",
  "tumor_size": 4, "estrogen_status": "Positive",
  "progesterone_status": "Positive",
  "regional_node_examined": 24, "regional_node_positive": 1
}
```

Response:
```json
{"prediction": "Alive", "probability_dead": 0.3284, "model_version": "1"}
```

Inputs are validated (allowed categories, value ranges, positive nodes ≤ examined
nodes); invalid requests get HTTP `422` and never reach the model. Every valid
prediction is logged to `predictions.db` (SQLite, git-ignored) for drift monitoring.

## Run with Docker
The whole system (MLflow server, training, API) starts with one command. Requires
Docker Desktop and `data/raw/Breast_Cancer.csv`; stop any local `mlflow server` or
`uvicorn` first, because the containers use ports 5000 and 8000.
```powershell
docker compose up --build
```

| Service | What it does |
|---|---|
| `mlflow` | Tracking server + model registry at http://localhost:5000 (data kept in the `mlflow-data` volume) |
| `train` | Starts when MLflow is healthy, trains the model, registers `breast-cancer-survival@champion`, then exits |
| `api` | Starts after training succeeds; API at http://localhost:8000 (`/docs`, `/health`, `/predict`); prediction log kept in the `api-logs` volume |

All three services use one image built from the `Dockerfile` (Python 3.11-slim,
non-root user). The dataset is mounted read-only into `train`, never copied into the image.

Stop with `Ctrl+C`, then `docker compose down` (volumes are kept; add `-v` to delete them).
Note: each `up` retrains and registers a new model version.

`requirements.txt` marks the Windows-only packages `pywin32` and `waitress` with
`; sys_platform == "win32"` so the Linux image builds. Re-add these markers if you
regenerate the file with `pip freeze`.

## Tests
Run from the project root (needs `data/raw/Breast_Cancer.csv`; MLflow does not need to be running):
```powershell
python -m pytest
```

41 tests in three files:

| File | Checks |
|---|---|
| `tests/test_data.py` | Cleaning rules, the 13 features, no leakage columns, target balance |
| `tests/test_model.py` | **Quality gate** (recall ≥ 0.55, ROC-AUC ≥ 0.70), reproducibility, sensible risk ordering |
| `tests/test_api.py` | `/health` and `/predict` (200 / 422 / 503), prediction logging, API matches the model, schema matches the training data |

API tests replace MLflow and the database with test doubles, so they run offline
and never touch the real `predictions.db`.

## Status
- [x] Phase 1: Repository setup
- [x] Phase 2: Training pipeline
- [x] Phase 3: MLflow tracking
- [x] Phase 4: FastAPI service
- [x] Phase 5: Automated tests (pytest)
- [x] Phase 6: Docker and docker-compose
- [ ] Phase 7: Jenkins CI/CD