"""Export deterministic public OpenAPI and derived TypeScript; no model startup."""
import argparse
import json
from pathlib import Path
from depi_ml.api.main import create_app
from depi_ml.infrastructure.config.api_settings import ApiSettings

def typescript_type(schema):
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "const" in schema:
        return json.dumps(schema["const"], ensure_ascii=False)
    if "enum" in schema:
        return " | ".join(json.dumps(v, ensure_ascii=False) for v in schema["enum"])
    for union in ("oneOf", "anyOf"):
        if union in schema:
            return " | ".join(typescript_type(v) for v in schema[union])
    kind = schema.get("type")
    if kind == "null":
        return "null"
    if kind in ("integer", "number"):
        return "number"
    if kind in ("string", "boolean"):
        return kind
    if kind == "array":
        if "prefixItems" in schema:
            return "[" + ", ".join(typescript_type(v) for v in schema["prefixItems"]) + "]"
        return "Array<" + typescript_type(schema.get("items", {})) + ">"
    if kind == "object":
        required = schema.get("required", [])
        return "{\n" + "".join("  " + json.dumps(k) + ("" if k in required else "?") + ": " +
            typescript_type(v) + ";\n" for k, v in sorted(schema.get("properties", {}).items())) + "}"
    return "unknown"

def render_types(schema):
    header = """// Generated from ml-internal-v1.openapi.json by scripts/export_ml_contract.py.
// Reference only; no Fastify client or runtime validation is implemented here.
// Validate JSON against OpenAPI: no extras, finite numbers, explicit required
// nulls, integer/range constraints and cross-feature consistency. In JS use
// Number.isSafeInteger for integer fields and Number.isFinite for scores/floats.
// Header X-Service-API-Key belongs only in Fastify; never send it to a browser.
// Correlate opaque X-Request-ID (1..64 ASCII [A-Za-z0-9._-]); never personal IDs.
export type MlModelKey = "random_forest_full_v1" | "xgboost_full_v1";
export type FeatureVersion = "BOOKING_BASELINE_V1";
export type MlRequestHeaders = { "X-Request-ID"?: string; "X-Service-API-Key": string };
"""
    return header + "\n".join("export type " + name + " = " + typescript_type(component) + ";\n"
                             for name, component in sorted(schema["components"]["schemas"].items())) + """
export type BatchRequest = BatchSingleRequest | BatchCompareRequest;
export type BatchResponse = BatchSingleResponse | BatchCompareResponse;
export type ShapExplanation = ProbabilityExplanation | LogOddsExplanation;
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/contracts"))
    args = parser.parse_args()
    schema = create_app(ApiSettings(env="development", expose_docs=True)).openapi()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ml-internal-v1.openapi.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "ml-internal-v1.types.ts").write_text(render_types(schema), encoding="utf-8")
    print("Public OpenAPI and TypeScript exported; no models, DB or secrets.")

if __name__ == "__main__":
    main()
