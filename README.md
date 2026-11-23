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
src/         training and prediction code
api/         FastAPI service
tests/       pytest tests
monitoring/  Evidently drift reports
data/raw/    dataset (git-ignored)
```

## Train
```powershell
python -m src.train
```

Baseline results (held-out 20% test set, Dead = positive class):

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.626 | 0.280 | 0.387 | 0.731 |
| Random Forest | 0.537 | 0.297 | 0.383 | 0.713 |
| Gradient Boosting | 0.415 | 0.282 | 0.336 | 0.689 |

## Status
- [x] Phase 1: Repository setup
- [x] Phase 2: Training pipeline
- [ ] Phase 3: MLflow tracking