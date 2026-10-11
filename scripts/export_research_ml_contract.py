"""Contrato v2 separado; nunca sobrescribe contratos v1."""

import argparse
import json
from pathlib import Path

from depi_ml.api.main import create_app
from depi_ml.api.openapi_versions import versioned_openapi
from depi_ml.infrastructure.config.api_settings import ApiSettings
if __package__:
    from .export_ml_contract import typescript_type
else:
    from export_ml_contract import typescript_type


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("docs/contracts"))
    args=parser.parse_args()
    schema=versioned_openapi(create_app(ApiSettings()),"v2")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Deterministic generated files; only the explicitly named new v2 outputs.
    (args.output_dir/"ml-internal-v2.openapi.json").write_text(json.dumps(schema,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    types="// Generated v2 research contract; operational_use_allowed=false.\n"
    types+='\n'.join("export type "+name+" = "+typescript_type(component)+";\n"
                      for name,component in sorted(schema["components"]["schemas"].items()))
    (args.output_dir/"ml-internal-v2.types.ts").write_text(types)
    print("v2 OpenAPI and types exported; v1 files unchanged.")


if __name__=="__main__":
    main()
