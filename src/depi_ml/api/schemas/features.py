"""Validación JSON estricta; nombres/orden derivados del contrato existente."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from depi_ml.datasets.schema import PREDICTOR_COLUMNS

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
FEATURE_TYPES = {
    "age_at_booking": Annotated[int, Field(ge=16, le=95)] | None,
    "client_sex": Literal["M", "F", "X"] | None,
    "booking_lead_days": Annotated[float, Field(gt=0, allow_inf_nan=False)],
    "appointment_month": Annotated[int, Field(ge=1, le=12)],
    "appointment_weekday": Annotated[int, Field(ge=1, le=7)],
    "appointment_hour": Annotated[int, Field(ge=0, le=23)],
    "duration_minutes": NonNegativeInt | None,
    "is_fwa": bool | None,
    "clinic_id": PositiveInt,
    "scheduled_service_lines": NonNegativeInt,
    "distinct_body_areas": NonNegativeInt,
    "single_body_area_id": PositiveInt | None,
    "has_medical_evaluation": bool | None,
    "has_type4_service": bool | None,
    "previous_attended": NonNegativeInt,
    "previous_no_show": NonNegativeInt,
    "previous_no_show_rate": Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None,
    "days_since_previous_attended": NonNegativeFloat | None,
    "days_since_previous_no_show": NonNegativeFloat | None,
}
# Fail on schema drift rather than maintain another ordered feature list.
if set(FEATURE_TYPES) != set(PREDICTOR_COLUMNS):
    raise RuntimeError("El esquema HTTP requiere revisión del contrato predictor.")


class FeatureValidation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def coherent_history_and_services(self):
        if self.distinct_body_areas > self.scheduled_service_lines:
            raise ValueError("inconsistent_service_counts")
        if (self.distinct_body_areas == 1) != (self.single_body_area_id is not None):
            raise ValueError("inconsistent_single_body_area")
        total = self.previous_attended + self.previous_no_show
        if total == 0:
            if self.previous_no_show_rate is not None:
                raise ValueError("history_rate_requires_events")
        elif self.previous_no_show_rate is None or abs(self.previous_no_show_rate - self.previous_no_show / total) > 1e-6:
            raise ValueError("inconsistent_history_rate")
        if self.previous_attended == 0 and self.days_since_previous_attended is not None:
            raise ValueError("attendance_recency_requires_events")
        if self.previous_no_show == 0 and self.days_since_previous_no_show is not None:
            raise ValueError("no_show_recency_requires_events")
        return self


PredictionFeatures = create_model("PredictionFeatures", __base__=FeatureValidation,
                                 **{name: (FEATURE_TYPES[name], ...) for name in PREDICTOR_COLUMNS})
