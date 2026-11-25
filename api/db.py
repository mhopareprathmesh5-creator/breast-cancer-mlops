"""Prediction log: every /predict request is saved to a SQL database.

Uses SQLAlchemy (ORM) with SQLite by default. Evidently (Phase 8) reads this
table to compare production inputs with the training data (drift detection).
"""
import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.data import PROJECT_ROOT

# Database location as a connection URL. Read from an environment variable so
# Docker can point it elsewhere (e.g. a mounted volume); defaults to a file in
# the project root. Switching to Postgres/MySQL later only changes this URL,
# e.g. "postgresql://user:pass@host/db"; the rest of the code stays the same.
# (predictions.db is git-ignored by the *.db rule.)
DATABASE_URL = os.getenv(
    "PREDICTIONS_DB_URL",
    f"sqlite:///{(PROJECT_ROOT / 'predictions.db').as_posix()}",
)

# Engine = the connection to the database (opened lazily, reused).
engine = create_engine(DATABASE_URL)

# Session factory: each request opens a short-lived session (a "unit of work"),
# adds rows, commits, and closes it.
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    """Parent class for all table classes (SQLAlchemy 2.x style)."""


class PredictionLog(Base):
    """One row per prediction. Each attribute becomes a column.

    Mapped[int] / Mapped[str] / Mapped[float] tell SQLAlchemy the column type
    from the Python type hint. Column names match the API's snake_case fields,
    so a row can be built straight from PatientFeatures.model_dump().
    """

    __tablename__ = "predictions"

    # Primary key: auto-incrementing row id (1, 2, 3, ...).
    id: Mapped[int] = mapped_column(primary_key=True)

    # When the prediction was made, in UTC (no time-zone confusion). Lets
    # Evidently compare e.g. last week's patients with the training data.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # --- The 13 input features (same names as api/schemas.py) ---
    age: Mapped[int]
    tumor_size: Mapped[int]
    regional_node_examined: Mapped[int]
    regional_node_positive: Mapped[int]
    race: Mapped[str]
    marital_status: Mapped[str]
    t_stage: Mapped[str]
    n_stage: Mapped[str]
    sixth_stage: Mapped[str]
    differentiate: Mapped[str]
    a_stage: Mapped[str]
    estrogen_status: Mapped[str]
    progesterone_status: Mapped[str]

    # --- The model's output ---
    prediction: Mapped[str]  # "Alive" or "Dead"
    probability_dead: Mapped[float]  # 0-1 risk score
    model_version: Mapped[str]  # which registry version produced it


def init_db() -> None:
    """Create the table if it doesn't exist yet (safe to call on every startup)."""
    Base.metadata.create_all(engine)