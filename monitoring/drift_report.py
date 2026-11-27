"""Data drift report: production inputs (the API's prediction log) vs the
training data.

Drift = the patients the API now receives look different from the patients
the model learned from. True outcomes take years to arrive, so we can't
measure accuracy in production; checking the INPUTS is the early warning.

Runs inside Docker, next to the API's log volume (see docker-compose.yml):
    docker compose run --rm monitor
Writes reports/drift_report.html and prints a short summary.
"""
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from api.db import engine  # the same database the API logs to
from api.schemas import PatientFeatures
from scripts.smoke_check import PATIENT as SMOKE_PATIENT
from src.compare_models import RANDOM_STATE, TEST_SIZE
from src.data import (
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    PROJECT_ROOT,
    load_data,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.html"

# Below this many rows, statistical tests can't say much (rule of thumb).
MIN_ROWS = 30

# API/log field name -> training column name, e.g. "t_stage" -> "T Stage".
FIELD_TO_COLUMN = {
    name: field.serialization_alias for name, field in PatientFeatures.model_fields.items()
}


def reference_data() -> pd.DataFrame:
    """What the model learned from: the 80% training split."""
    X, y = load_data()
    X_train, _, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train


def current_data() -> tuple[pd.DataFrame, int]:
    """What the API has received: the prediction log, without smoke-test rows.

    Every deployment's smoke test sends the same patient; those rows are
    checks, not real patients, so they are excluded.
    Returns (rows with training column names, number of smoke rows removed).
    """
    log = pd.read_sql_table("predictions", engine)

    smoke_fields = list(SMOKE_PATIENT)
    is_smoke = (log[smoke_fields] == pd.Series(SMOKE_PATIENT)).all(axis=1)

    real = log.loc[~is_smoke, list(FIELD_TO_COLUMN)].rename(columns=FIELD_TO_COLUMN)
    return real, int(is_smoke.sum())


def main() -> None:
    reference = reference_data()
    current, smoke_rows = current_data()
    print(f"Reference (training data): {len(reference)} rows")
    print(f"Current (prediction log):  {len(current)} rows "
          f"({smoke_rows} smoke-test rows excluded)")

    if current.empty:
        print("No real predictions logged yet: nothing to compare.")
        return
    if len(current) < MIN_ROWS:
        print(f"WARNING: fewer than {MIN_ROWS} rows; the result is not "
              "statistically reliable yet.")

    # Tell Evidently which columns are numbers and which are categories, so it
    # picks the right statistical test for each.
    definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=list(ORDINAL_FEATURES) + NOMINAL_FEATURES,
    )
    snapshot = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )

    REPORT_PATH.parent.mkdir(exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))

    # Short summary in the terminal; full details are in the HTML report.
    for metric in snapshot.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            print(f"Drifted columns: {metric['value']}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()