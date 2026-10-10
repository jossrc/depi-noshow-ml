from pydantic import BaseModel


class ErrorDetail(BaseModel):
    field: str
    reason: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail]


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str


ERROR_RESPONSES = {status: {"model": ErrorResponse} for status in (400, 401, 404, 409, 413, 422, 429, 500, 503)}
