"""Production training: train the chosen model, track it and register it.

Model SELECTION happens in src/compare_models.py (Logistic Regression won).
This script trains only Logistic Regression and:
  1. cross-validates it on the training set (a stable estimate of its scores),
  2. trains it on the full training set and scores it on the held-out test set,
  3. logs parameters, metrics and the trained pipeline to MLflow,
  4. registers the pipeline in the MLflow Model Registry and points the
     @champion alias at it. The API loads models:/breast-cancer-survival@champion.

Run from the project root (with `mlflow server` running in another terminal):
    python -m src.train
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os  # read the MLflow server address from an environment variable

import mlflow  # experiment tracking + model registry
import mlflow.sklearn  # save/load scikit-learn models in MLflow's format
from mlflow import MlflowClient  # lower-level API, used to set the alias
from mlflow.models import infer_signature  # records the expected input/output
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data import load_data

# Reuse the exact preprocessing, evaluation and settings from the comparison,
# so the production model is built the same way as the one that won.
from src.compare_models import RANDOM_STATE, TEST_SIZE, build_pipeline, evaluate

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CV_FOLDS = 5  # number of cross-validation folds

# Where the MLflow server is. Read from an environment variable so the same
# code works locally, in Docker and in Jenkins (each sets its own URL); falls
# back to the local server started with `mlflow server`.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

EXPERIMENT_NAME = "breast-cancer-survival"  # groups the runs in the MLflow UI
REGISTERED_MODEL_NAME = "breast-cancer-survival"  # name in the Model Registry
CHAMPION_ALIAS = "champion"  # the alias the API loads

# Metrics computed in cross-validation (scikit-learn's built-in scorer names).
# All treat Dead (= 1) as the positive class.
SCORING = {"recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc"}


def build_model() -> LogisticRegression:
    """The production model: same settings as in src/compare_models.py.

    class_weight="balanced" makes mistakes on the rare Dead class (15%) count
    more, so the model doesn't just predict "Alive" for everyone.
    """
    return LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )


def cross_validate_pipeline(pipeline, X, y) -> dict:
    """5-fold cross-validation on the training set.

    The training set is cut into 5 parts; the pipeline is trained on 4 and
    scored on the 5th, five times, so every patient is scored once. The mean
    is more reliable than one test split (only ~123 Dead patients); the std
    shows how much the score moves between folds.
    StratifiedKFold keeps the 15% Dead rate in every fold; shuffle + seed make
    the folds random but repeatable.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)

    # scores["test_recall"] holds 5 numbers (one per fold) -> keep mean and std.
    metrics = {}
    for metric in SCORING:
        metrics[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
        metrics[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
    return metrics


def main() -> None:
    # Connect to the MLflow server and choose (or create) the experiment.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Same cleaned data and the same stratified 80/20 split as the comparison.
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = build_model()

    # Everything inside this block is recorded as one MLflow "run".
    with mlflow.start_run(run_name="logistic_regression"):
        pipeline = build_pipeline(model)

        # 1. Cross-validate on the training set.
        cv_metrics = cross_validate_pipeline(pipeline, X_train, y_train)

        # 2. Train on the whole training set, score once on the test set.
        pipeline.fit(X_train, y_train)
        test_metrics = {
            f"test_{k}": v for k, v in evaluate(pipeline, X_test, y_test).items()
        }

        # 3a. Parameters: the settings that produced this run.
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
            }
        )
        # Every hyperparameter of the model, prefixed so they're easy to find.
        mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})

        # 3b. Metrics: the scores this run achieved.
        mlflow.log_metrics({**cv_metrics, **test_metrics})

        # 3c + 4. Save the trained pipeline (preprocessing + model) and register it.
        # signature: expected input columns/types and output type; MLflow
        #   checks incoming data against it when the model is used.
        # input_example: 3 real rows, shown in the UI as a usage example.
        # registered_model_name: also registers it, creating a new version
        #   (v1, v2, ...) of "breast-cancer-survival" on every run.
        # MLflow 3 saves it with skops (a safe format, not pickle); Logistic
        # Regression needs no extra trusted types. Tree models would need
        # skops_trusted_types=[...] here.
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train.head(3),
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    # Print the scores.
    for name, value in {**cv_metrics, **test_metrics}.items():
        print(f"{name:20s} {value:.3f}")

    # Point the "champion" alias at the new version. The API loads
    # models:/breast-cancer-survival@champion, so moving the alias deploys a
    # model, and moving it back rolls back, with no code changes.
    version = model_info.registered_model_version
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version
    )
    print(f"\nRegistered {REGISTERED_MODEL_NAME} v{version} @{CHAMPION_ALIAS}")


if __name__ == "__main__":
    main()
