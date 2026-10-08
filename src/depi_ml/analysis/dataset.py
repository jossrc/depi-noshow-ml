"""Lectura estricta del contrato y verificación del archivo original."""

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from depi_ml.datasets.schema import COLUMNS, EXPORT_COLUMNS
from depi_ml.errors import Phase2Error


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class LocalDataset:
    frame: pd.DataFrame
    manifest: dict
    source: Path
    sha256: str
    manifest_source: Path | None = None
    manifest_sha256: str | None = None


def load_dataset(csv_path: Path, manifest_path: Path | None = None) -> LocalDataset:
    csv_path = Path(csv_path)
    manifest_path = manifest_path or csv_path.with_name(csv_path.stem + "_manifest.json")
    if not csv_path.is_file() or not manifest_path.is_file():
        raise Phase2Error("Falta el CSV local o su manifest; no se inventan resultados.")
    try:
        manifest_sha = digest(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sha = digest(csv_path)
        if not isinstance(manifest, dict):
            raise Phase2Error("Manifest inválido.")
        if manifest.get("sha256") != sha:
            raise Phase2Error("SHA-256 del CSV distinto del manifest.")
        if manifest.get("columns") != EXPORT_COLUMNS or manifest.get("column_count") != len(EXPORT_COLUMNS):
            raise Phase2Error("El manifest no coincide con el contrato central.")
        if manifest.get("size_bytes") != csv_path.stat().st_size:
            raise Phase2Error("Tamaño del CSV distinto del manifest.")
        # pandas puede admitir filas cortas: verificar también registros CSV lógicos.
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            if next(reader, None) != EXPORT_COLUMNS:
                raise Phase2Error("Columnas CSV distintas del contrato central.")
            count = 0
            for row in reader:
                if len(row) != len(EXPORT_COLUMNS):
                    raise Phase2Error("Registro CSV con cantidad de columnas incorrecta.")
                count += 1
        if count == 0 or count != manifest.get("row_count"):
            raise Phase2Error("Cantidad de registros distinta del manifest o dataset vacío.")
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
        for column in COLUMNS:
            name, kind = column.name, column.pg_type
            values = frame[name]
            if kind == "timestamptz":
                if (~values.dropna().str.contains(r"(?:Z|[+-]\d{2}(?::?\d{2})?)$", regex=True)).any():
                    raise Phase2Error(f"Timestamp sin zona horaria en {name}.")
                parsed = pd.to_datetime(values, utc=True, format="ISO8601", errors="coerce")
            elif kind == "bool":
                parsed = values.map({"t": 1., "f": 0., "true": 1., "false": 0., "1": 1., "0": 0.})
            elif kind.startswith("int") or kind == "float8":
                parsed = pd.to_numeric(values, errors="coerce")
                if kind.startswith("int") and (parsed.dropna() % 1 != 0).any():
                    raise Phase2Error(f"Entero inválido en {name}.")
                if not np.isfinite(parsed.dropna()).all():
                    raise Phase2Error(f"Número no finito en {name}.")
            else:
                continue
            if (values.notna() & parsed.isna()).any():
                raise Phase2Error(f"Tipo inválido en {name}.")
            frame[name] = parsed
        if len(frame) != count or digest(csv_path) != sha or digest(manifest_path) != manifest_sha:
            raise Phase2Error("El archivo cambió durante la lectura.")
        versions = sorted(frame.feature_version.dropna().unique().tolist())
        if versions != sorted(manifest.get("feature_versions", [])):
            raise Phase2Error("Versiones de características distintas del manifest.")
        return LocalDataset(frame, manifest, csv_path.resolve(), sha, manifest_path.resolve(), manifest_sha)
    except (pd.errors.ParserError, UnicodeError, csv.Error, json.JSONDecodeError):
        raise Phase2Error("CSV o manifest con formato inválido.") from None


def new_output(path: Path, dataset: LocalDataset) -> Path:
    """No sobrescribir reportes previos ni escribir junto al CSV/manifest."""
    path = Path(path).resolve()
    if path == dataset.source.parent or dataset.source.parent in path.parents:
        raise Phase2Error("La salida debe estar fuera de la carpeta del CSV fuente.")
    if path.exists():
        raise Phase2Error("La carpeta de salida ya existe; elige otra para conservar artefactos.")
    path.mkdir(parents=True)
    return path


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
