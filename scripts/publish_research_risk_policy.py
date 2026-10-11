"""Publicar una versión research provisional nueva tras verificar identidad e integridad.

Este paso no aprueba uso operativo, no carga CSV y no deserializa modelos.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerifier
from depi_ml.infrastructure.ml.model_registry import ModelRegistry
from depi_ml.infrastructure.ml.research_risk_registry import ResearchRiskPolicyRegistry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--policy-sha256",required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path.cwd().resolve()
    if not args.source.resolve().is_relative_to(root/"reports"):
        parser.error("Fuente de una ejecución local bajo reports requerida.")
    output=args.output.resolve()
    if not output.is_relative_to(root/"policies") or output.exists():
        parser.error("Publicación nueva dentro de policies/ requerida.")
    verified=ArtifactVerifier(args.reference,args.manifest_sha256).verify()
    registry=ModelRegistry(SimpleNamespace(info=a.info) for a in verified)
    settings=ApiSettings(research_risk_policy_path=args.source,research_risk_policy_sha256=args.policy_sha256,
                         model_reference_dir=args.reference,model_manifest_sha256=args.manifest_sha256)
    policies=ResearchRiskPolicyRegistry.load(settings,registry)
    raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.policy_sha256:
        raise ValueError("Policy changed before publication")
    output.mkdir(parents=True,exist_ok=False)
    (output/"risk_policy.json").write_bytes(raw)
    publication={"status":"RESEARCH_PROVISIONAL","operational_use_allowed":False,"approved_by":None,
                 "policy_file_sha256":args.policy_sha256,"reference_manifest_sha256":args.manifest_sha256,
                 "checks":["trusted_original_models_verified_without_deserialization","both_model_identities_match",
                           "policy_content_hashes_match","nondegenerate_thresholds","cohort_support_and_monotonicity"],
                 "policies":[{"model_key":p.model_key,"policy_hash":p.policy_hash_sha256} for p in policies.list_policies()],
                 "publication_meaning":"technical_research_release_only_not_operational_or_scientific_approval"}
    (output/"publication.json").write_text(json.dumps(publication,indent=2)+'\n')
    print(json.dumps(publication,indent=2))


if __name__=="__main__":
    main()
