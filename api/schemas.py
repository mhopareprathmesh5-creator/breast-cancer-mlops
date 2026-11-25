"""Request and response formats for the API (Pydantic models).

FastAPI uses these to:
  - validate every request automatically (bad input -> HTTP 422, before the
    model is ever called),
  - convert JSON <-> Python objects,
  - generate the interactive documentation at /docs.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PatientFeatures(BaseModel):
    """The 13 features the model needs for one patient.

    JSON field names are snake_case (easy for clients to type). Each field's
    serialization_alias is the column name the model was trained on, so
    model_dump(by_alias=True) produces exactly the columns the Pipeline expects.
    """

    # extra="forbid": reject unknown fields (e.g. a typo like "tumour_size")
    # instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid")

    # --- Numbers: ge/le = allowed range (greater-or-equal / less-or-equal) ---
    # Ranges are wider than the training data (age 30-69) but block impossible
    # values. Unusual-but-valid inputs are what drift monitoring (Phase 8) watches.
    age: int = Field(ge=18, le=100, serialization_alias="Age")
    tumor_size: int = Field(ge=1, le=200, serialization_alias="Tumor Size")  # mm
    regional_node_examined: int = Field(
        ge=1, le=100, serialization_alias="Regional Node Examined"
    )
    regional_node_positive: int = Field(
        ge=1, le=100, serialization_alias="Reginol Node Positive"  # dataset typo kept
    )

    # --- Categories: Literal = only these exact values are accepted ---
    # They match the cleaned training data, so the model never sees a value it
    # wasn't trained on (Phase 5 adds a test that checks they stay in sync).
    race: Literal["White", "Black", "Other"] = Field(serialization_alias="Race")
    marital_status: Literal[
        "Married", "Single", "Divorced", "Widowed", "Separated"
    ] = Field(serialization_alias="Marital Status")
    t_stage: Literal["T1", "T2", "T3", "T4"] = Field(serialization_alias="T Stage")
    n_stage: Literal["N1", "N2", "N3"] = Field(serialization_alias="N Stage")
    sixth_stage: Literal["IIA", "IIB", "IIIA", "IIIB", "IIIC"] = Field(
        serialization_alias="6th Stage"
    )
    differentiate: Literal[
        "Well differentiated",
        "Moderately differentiated",
        "Poorly differentiated",
        "Undifferentiated",
    ] = Field(serialization_alias="differentiate")
    a_stage: Literal["Regional", "Distant"] = Field(serialization_alias="A Stage")
    estrogen_status: Literal["Positive", "Negative"] = Field(
        serialization_alias="Estrogen Status"
    )
    progesterone_status: Literal["Positive", "Negative"] = Field(
        serialization_alias="Progesterone Status"
    )

    # --- Rule across two fields: runs after the single-field checks pass ---
    @model_validator(mode="after")
    def positive_not_more_than_examined(self):
        # You can't find cancer in more nodes than were examined (EDA confirmed
        # this holds in every training row).
        if self.regional_node_positive > self.regional_node_examined:
            raise ValueError(
                "regional_node_positive cannot be greater than regional_node_examined"
            )
        return self


class PredictionResponse(BaseModel):
    """What /predict returns."""

    prediction: Literal["Alive", "Dead"]
    probability_dead: float  # 0-1: the model's estimated risk of death
    model_version: str  # which registry version answered (traceability)