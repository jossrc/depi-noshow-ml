from typing import Annotated, Literal
from pydantic import Field

from depi_ml.api.schemas.features import PredictionFeatures
from depi_ml.api.schemas.prediction import StrictObject, FeatureContext, PredictionResponse, CompareResponse


class BatchItem(StrictObject):
    features: PredictionFeatures


class BatchRequestBase(StrictObject):
    feature_version: Annotated[str, Field(min_length=1, max_length=64)]
    feature_context: FeatureContext
    items: Annotated[list[BatchItem], Field(min_length=1, max_length=64)]


class BatchSingleRequest(BatchRequestBase):
    mode: Literal["single"]
    model_key: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")] | None = None


class BatchCompareRequest(BatchRequestBase):
    mode: Literal["compare"]


BatchRequest = Annotated[BatchSingleRequest | BatchCompareRequest, Field(discriminator="mode")]


class SingleBatchItemResult(StrictObject):
    index: Annotated[int, Field(ge=0)]
    prediction: PredictionResponse


class CompareBatchItemResult(StrictObject):
    index: Annotated[int, Field(ge=0)]
    prediction: CompareResponse


class BatchResponseBase(StrictObject):
    api_version: Literal["v1"] = "v1"
    feature_version: Literal["BOOKING_BASELINE_V1"] = "BOOKING_BASELINE_V1"
    count: Annotated[int, Field(ge=1, le=64)]
    risk_level: None = None
    risk_policy_status: Literal["pending_validation"] = "pending_validation"
    validation_status: Literal["EXPLORATORY_NOT_VALIDATED"] = "EXPLORATORY_NOT_VALIDATED"
    calibration_status: Literal["not_operationally_validated"] = "not_operationally_validated"
    operational_use_allowed: Literal[False] = False
    request_id: str


class BatchSingleResponse(BatchResponseBase):
    mode: Literal["single"]
    results: Annotated[list[SingleBatchItemResult], Field(min_length=1, max_length=64)]


class BatchCompareResponse(BatchResponseBase):
    mode: Literal["compare"]
    results: Annotated[list[CompareBatchItemResult], Field(min_length=1, max_length=64)]


BatchResponse = Annotated[BatchSingleResponse | BatchCompareResponse, Field(discriminator="mode")]
