# CLAUDE.md — Breast Cancer Survival Classification (MLOps learning project)

## Purpose
Rebuild a resume ML project cleanly so the user deeply understands each tool for interviews:
**Git/GitHub, Docker, Jenkins, pytest, Evidently, MLflow** (+ Python, FastAPI, scikit-learn).

## How we work (IMPORTANT)
- **Claude guides, the user performs.** Explain *what* and *why* for each step, give the commands/code, and let the user run them.
- Do not write project code into files unless the user explicitly asks. The user reports output/errors back; Claude helps debug.
- Every technology choice must have a stated reason (interview-ready).
- End each phase with checkpoint interview questions.
- Environment: Windows 11, PowerShell, VS Code. Project path: `D:\breast-cancer-mlops`.

## Decisions and reasons
| Decision | Choice | Why |
|---|---|---|
| Dataset | SEER Breast Cancer (Kaggle: reihanenamdari/breast-cancer, `Breast_Cancer.csv`, ~4,024 rows) | Larger than Wisconsin (569 rows); mixed numeric + categorical columns → real preprocessing; ~15% "Dead" class → class-imbalance handling; enough rows to hold out "production" data for drift checks. |
| Task | Binary classification of `Status` (Alive vs Dead) | Survival prediction from clinical features. |
| ML vs DL | Classical ML | Small tabular data: trees/linear models match or beat neural nets, are more explainable (important in medicine), and train in seconds, keeping CI fast. |
| Models | Logistic Regression (baseline), Random Forest, Gradient Boosting, compared in MLflow | Pick the winner from evidence; Gradient Boosting is usually strongest on mixed, imbalanced tabular data. |
| Main metric | Recall on the "Dead" class (+ F1, ROC-AUC) | Missing a high-risk patient (false negative) is costlier than a false alarm; accuracy hides imbalance. |
| Python | 3.11 (via `py -3.11 -m venv venv`) | 3.10 reached end-of-life Oct 2026; Docker image `python:3.11-slim` matches local and CI. |
| API | FastAPI only (Flask dropped) | Pydantic validation, auto Swagger docs at `/docs`, async, built-in TestClient. Two frameworks added nothing. |
| Storage | SQLite via SQLAlchemy (MySQL dropped) | Logs predictions for Evidently; no server needed; swapping to MySQL/Postgres is a connection-string change. |
| Tracking | MLflow (experiments + Model Registry) | Logs params, metrics and models; the API loads the registered "champion" model instead of a hard-coded `.pkl`. |
| Testing | pytest | Data and preprocessing tests, a model-quality threshold test, API tests with TestClient. |
| Containers | Docker + docker-compose | Same environment everywhere; compose runs the API and the MLflow server together. |
| CI/CD | Jenkins (in Docker) + `Jenkinsfile` | Pipeline as code: checkout → install → test → train → build image → smoke test → deploy. |
| Monitoring | Evidently | Data drift reports comparing training (reference) data with logged production data. |
| Git workflow | Feature branch per phase → PR → merge to `main` | Keeps `main` working; professional workflow. |
| Data in Git | Not committed (`data/` in `.gitignore`) | Git is for code, not datasets; licensing concerns; DVC is the proper tool for data versioning. |

## Planned structure
```
breast-cancer-mlops/
├── src/            # data.py, train.py, predict.py
├── api/            # main.py (FastAPI), schemas.py, db.py
├── tests/          # test_data.py, test_model.py, test_api.py
├── monitoring/     # drift_report.py (Evidently)
├── data/raw/       # Breast_Cancer.csv (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
├── requirements.txt
└── README.md
```

## Roadmap
| Phase | Build | Tool |
|---|---|---|
| 0 | Install and verify tools | — |
| 1 | Repo skeleton, venv, `.gitignore`, first commit and push | Git, GitHub |
| 2 | Data loading, EDA, sklearn Pipeline training script | scikit-learn |
| 3 | Experiment tracking and model registry | MLflow |
| 4 | `/predict`, `/health` endpoints and the prediction log | FastAPI |
| 5 | Data, model and API tests | pytest |
| 6 | Dockerfile and docker-compose | Docker |
| 7 | Jenkins pipeline triggered from GitHub | Jenkins |
| 8 | Drift report on logged predictions | Evidently |
| 9 | Rewrite resume bullets and interview Q&A | — |

## Progress (update as phases complete)
- [x] Phase 0: Tools verified. Git 2.54, Docker 29.9, Compose v5.6, VS Code 1.146. Python 3.11 installed (old global 3.10.0 is still present; the venv uses 3.11).
- [x] Phase 1: Repo skeleton done. venv `venv/` (Python 3.11.9); folders with `__init__.py`; `.gitignore`; dataset in `data/raw/` (4,024 rows + header, 16 cols; `"T Stage "` has a trailing space → clean column names in code); README; first commit `a6b9c20` pushed to `origin/main` at https://github.com/mhopareprathmesh5-creator/breast-cancer-mlops.
- [ ] Phase 2: Next. Work on branch `phase-2-training`. Open question to discuss first: should `Survival Months` be a feature? (data leakage)
- [ ] Phase 3 to 9: not started.

## Notes
- Phase 1 `.gitignore` covers: `__pycache__/`, `*.pyc`, `venv/`, `.pytest_cache/`, `data/`, `mlruns/`, `mlartifacts/`, `*.db`, `reports/`, `.env`, `.vscode/`.
- If venv activation fails: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- Commit this file to the repo too; it doubles as design-decision documentation.
