from depi_ml.api.schemas.prediction import PredictionResponse, CompareResponse, Comparison


def prediction_response(result, request_id):
    return PredictionResponse(model_key=result.info.model_key, algorithm=result.info.algorithm,
                              model_version=result.info.model_version, model_hash=result.info.model_sha256[:12],
                              probability_no_show=result.probability_no_show, request_id=request_id)


def compare_response(results, request_id):
    scores = tuple(prediction_response(result, request_id) for result in results)
    return CompareResponse(results=scores, request_id=request_id,
                           comparison=Comparison(absolute_probability_difference=abs(
                               scores[0].probability_no_show - scores[1].probability_no_show)))
