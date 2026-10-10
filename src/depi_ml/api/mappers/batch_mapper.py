from depi_ml.api.mappers.prediction_mapper import prediction_response, compare_response
from depi_ml.api.schemas.batch import (BatchSingleResponse, BatchCompareResponse,
                                      SingleBatchItemResult, CompareBatchItemResult)


def batch_response(result, request_id):
    if result.mode == "single":
        rows = [SingleBatchItemResult(index=index, prediction=prediction_response(prediction, request_id))
                for index, prediction in enumerate(result.predictions)]
        return BatchSingleResponse(mode="single", count=len(rows), results=rows, request_id=request_id)
    rows = [CompareBatchItemResult(index=index, prediction=compare_response(prediction, request_id))
            for index, prediction in enumerate(result.predictions)]
    return BatchCompareResponse(mode="compare", count=len(rows), results=rows, request_id=request_id)
