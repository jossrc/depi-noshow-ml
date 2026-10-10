"""Registro inmutable de dos pipelines confiables, cargados una vez por lifespan."""

from dataclasses import dataclass
import logging
from types import MappingProxyType

from depi_ml.domain.errors import UnknownModel
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerificationError, ArtifactVerifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True, init=False)
class ModelRegistry:
    _predictors: object

    def __init__(self, predictors):
        entries = tuple(predictors)
        mapping = {predictor.info.model_key: predictor for predictor in entries}
        if len(mapping) != len(entries) or not set(mapping) <= set(MODEL_KEYS):
            raise ArtifactVerificationError("invalid_registry_keys")
        object.__setattr__(self, "_predictors", MappingProxyType(mapping))

    @property
    def ready(self):
        return set(self._predictors) == set(MODEL_KEYS)

    def get(self, model_key):
        try:
            return self._predictors[model_key]
        except KeyError:
            raise UnknownModel from None

    def list_models(self):
        return tuple(self._predictors[key].info for key in MODEL_KEYS if key in self._predictors)

    @classmethod
    def load(cls, settings):
        if settings.model_reference_dir is None or settings.model_manifest_sha256 is None:
            raise ArtifactVerificationError("missing_trusted_reference")
        try:
            verifier = ArtifactVerifier(settings.model_reference_dir, settings.model_manifest_sha256)
            # Verify BOTH references before deserializing either one.
            artifacts = verifier.verify()
            import joblib
            from depi_ml.infrastructure.ml.sklearn_predictor import RandomForestPredictor, XGBoostPredictor
            from depi_ml.infrastructure.ml.synthetic_probe import synthetic_features
            adapters = {"random_forest": RandomForestPredictor, "xgboost": XGBoostPredictor}
            predictors = []
            for artifact in artifacts:
                with verifier.verified_stream(artifact) as stream:
                    pipeline = joblib.load(stream)
                verify_pipeline(pipeline, artifact.info)
                predictor = adapters[artifact.info.algorithm](artifact.info, pipeline)
                predictor.predict_probability(synthetic_features())
                predictors.append(predictor)
                logger.info("event=model_loaded model_key=%s sha256=%s manifest_sha256=%s status=%s",
                            artifact.info.model_key, artifact.info.model_sha256,
                            artifact.info.manifest_sha256, artifact.info.validation_status)
            return cls(predictors)
        except ArtifactVerificationError:
            raise
        except Exception:
            raise ArtifactVerificationError("pipeline_load_or_probe_failed") from None


def verify_pipeline(pipeline, info):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted
    from xgboost import XGBClassifier
    from depi_ml.training.preprocessing import FeatureContract
    from depi_ml.infrastructure.ml.sklearn_predictor import positive_class_index

    if not isinstance(pipeline, Pipeline) or [name for name, _ in pipeline.steps] != ["contract", "preprocess", "model"]:
        raise ArtifactVerificationError("invalid_complete_pipeline")
    contract, preprocess, model = (pipeline.named_steps[name] for name in ("contract", "preprocess", "model"))
    expected = RandomForestClassifier if info.algorithm == "random_forest" else XGBClassifier
    if (not isinstance(contract, FeatureContract) or not isinstance(preprocess, ColumnTransformer)
            or not isinstance(model, expected) or contract.features_ != PREDICTOR_COLUMNS
            or tuple(info.features) != tuple(PREDICTOR_COLUMNS)
            or list(pipeline.feature_names_in_) != PREDICTOR_COLUMNS
            or list(preprocess.feature_names_in_) != PREDICTOR_COLUMNS
            or preprocess.n_features_in_ != len(PREDICTOR_COLUMNS)):
        raise ArtifactVerificationError("effective_pipeline_contract_mismatch")
    check_is_fitted(preprocess)
    check_is_fitted(model)
    if positive_class_index(pipeline) != positive_class_index(model):
        raise ArtifactVerificationError("ambiguous_pipeline_class_mapping")
