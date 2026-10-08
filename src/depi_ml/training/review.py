"""La evidencia externa no puede sustituirse por ordenar fechas o aceptar un flag."""

import json
from pathlib import Path

from depi_ml.analysis.audit import REVIEW_REQUIREMENTS, VERIFIED_REQUIREMENTS
from depi_ml.analysis.dataset import Phase2Error
from depi_ml.training.splits import utc_timestamp


def require_review(dataset, audit, review_path: Path | None, labels=None):
    if audit["hard_blockers"]:
        raise Phase2Error("Anomalías críticas del CSV: revisar audit.json; entrenamiento detenido.")
    if review_path is None or not review_path.is_file():
        raise Phase2Error("Falta revisión metodológica: ejecuta analyze-dataset y aporta --review con evidencia acreditada.")
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if not isinstance(review, dict):
            raise ValueError
        if review.get("dataset_sha256") != dataset.sha256 or review.get("feature_versions") != dataset.manifest["feature_versions"]:
            raise Phase2Error("La revisión no corresponde al hash/versiones del dataset.")
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            raise Phase2Error("Falta responsable de la revisión metodológica.")
        utc_timestamp(review.get("reviewed_at"))
        for name in REVIEW_REQUIREMENTS:
            check = review.get("checks", {}).get(name, {})
            accepted = {"verified"} if name in VERIFIED_REQUIREMENTS else {"verified", "acknowledged"}
            if check.get("status") not in accepted or not isinstance(check.get("evidence"), str) or len(check["evidence"].strip()) < 20:
                raise Phase2Error(f"Control metodológico pendiente: {name}.")
        if labels is None or review.get("label_availability_sha256") != labels.sha256 or review.get("label_availability_manifest_sha256") != labels.manifest_sha256:
            raise Phase2Error("La revisión debe vincularse al hash del auxiliar y su manifest; la demora global no es válida.")
        return review
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as error:
        if isinstance(error, Phase2Error):
            raise
        raise Phase2Error("Revisión metodológica inválida o incompleta.") from None
