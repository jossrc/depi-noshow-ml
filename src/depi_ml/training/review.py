"""Revisión definitiva y registro explícito de limitaciones exploratorias."""

import json
from pathlib import Path

from depi_ml.analysis.audit import REVIEW_REQUIREMENTS, VERIFIED_REQUIREMENTS
from depi_ml.analysis.dataset import Phase2Error, digest
from depi_ml.training.splits import utc_timestamp


EXPLORATORY_WARNING = "EXPLORATORY_NOT_VALIDATED"


def exploratory_review(dataset, audit, review_path, labels):
    """Conserva declaraciones originales; no acredita controles ni modifica plantillas."""
    if audit["hard_blockers"]:
        raise Phase2Error("Anomalías críticas del CSV: revisar audit.json; entrenamiento detenido.")
    snapshot = {
        "dataset_sha256": dataset.sha256, "feature_versions": dataset.manifest["feature_versions"],
        "label_availability_sha256": labels.sha256,
        "label_availability_manifest_sha256": labels.manifest_sha256,
        "reviewer": "", "reviewed_at": "",
        "checks": {name: {"status": "pending", "evidence": "", "instruction": instruction}
                   for name, instruction in REVIEW_REQUIREMENTS.items()},
    }
    review_sha = None
    if review_path is not None:
        path = Path(review_path)
        try:
            review_sha = digest(path)
            supplied = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(supplied, dict) or not isinstance(supplied.get("checks"), dict):
                raise ValueError
            for field in ["dataset_sha256", "feature_versions", "label_availability_sha256",
                          "label_availability_manifest_sha256"]:
                if supplied.get(field) != snapshot[field]:
                    raise Phase2Error("La revisión exploratoria no corresponde al dataset/auxiliar y sus hashes.")
            if any(not isinstance(check, dict) for check in supplied["checks"].values()):
                raise ValueError
            if digest(path) != review_sha:
                raise Phase2Error("La revisión cambió durante la lectura.")
            snapshot = supplied
        except (OSError, ValueError, TypeError) as error:
            if isinstance(error, Phase2Error):
                raise
            raise Phase2Error("Archivo de revisión exploratoria inexistente, inválido o incompleto.") from None
    signed = isinstance(snapshot.get("reviewer"), str) and bool(snapshot["reviewer"].strip())
    try:
        utc_timestamp(snapshot.get("reviewed_at"))
    except Phase2Error:
        signed = False
    pending = {}
    for name, instruction in REVIEW_REQUIREMENTS.items():
        check = snapshot["checks"].get(name, {})
        accepted = {"verified"} if name in VERIFIED_REQUIREMENTS else {"verified", "acknowledged"}
        evidence = check.get("evidence", "")
        if not signed or check.get("status") not in accepted or not isinstance(evidence, str) or len(evidence.strip()) < 20:
            pending[name] = {"reported_status": check.get("status", "pending"), "limitation": instruction}
    return {
        "status": EXPLORATORY_WARNING, "original_review_sha256": review_sha,
        "review_snapshot": snapshot, "pending_controls": pending,
        "limitations": ["Technical checks do not certify historical feature availability or scientific validity.",
                        "Validation results are exploratory; no test evaluation or definitive model selection is authorized.",
                        *[entry["limitation"] for entry in pending.values()]],
        "checks_automatically_verified": False,
    }


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
