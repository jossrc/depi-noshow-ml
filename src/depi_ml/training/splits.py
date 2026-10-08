"""Cortes por prediction_at: etiquetas maduras antes del período siguiente."""

from dataclasses import dataclass
import math

import pandas as pd

from depi_ml.analysis.dataset import Phase2Error


def utc_timestamp(value):
    try:
        time = pd.Timestamp(value)
        if pd.isna(time) or time.tzinfo is None:
            raise ValueError
        return time.tz_convert("UTC")
    except (ValueError, TypeError):
        raise Phase2Error("Cada corte requiere fecha ISO con zona horaria explícita.") from None


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: dict


def temporal_split(frame, validation_start, test_start, test_end, label_delay_hours, labels_observed_until):
    val, test, end, observed = map(utc_timestamp, [validation_start, test_start, test_end, labels_observed_until])
    if not val < test < end:
        raise Phase2Error("Se requiere validation_start < test_start < test_end.")
    if not isinstance(label_delay_hours, (int, float)) or isinstance(label_delay_hours, bool) or not math.isfinite(label_delay_hours) or label_delay_hours < 0:
        raise Phase2Error("La demora de etiqueta debe ser una cota finita no negativa acreditada por revisión.")
    if frame[["prediction_at", "appointment_at", "target"]].isna().any().any() or not frame.target.isin([0, 1]).all():
        raise Phase2Error("Fechas o etiquetas inválidas para dividir temporalmente.")
    if (frame.prediction_at >= frame.appointment_at).any():
        raise Phase2Error("Reserva posterior o simultánea a la cita.")
    try:
        known = frame.appointment_at + pd.Timedelta(hours=label_delay_hours)
    except (OverflowError, ValueError):
        raise Phase2Error("Demora de etiqueta fuera del rango temporal soportado.") from None
    masks = {
        "train": frame.prediction_at < val,
        "validation": (frame.prediction_at >= val) & (frame.prediction_at < test),
        "test": (frame.prediction_at >= test) & (frame.prediction_at < end),
    }
    cutoffs = {"train": min(val, observed), "validation": min(test, observed), "test": observed}
    partitions, details = {}, {}
    for name, mask in masks.items():
        retained = mask & (known < cutoffs[name])
        subset = frame.loc[retained].sort_values("prediction_at", kind="stable").copy()
        if subset.empty or subset.target.nunique() != 2:
            raise Phase2Error(f"Partición {name} vacía o con una sola clase tras purgar etiquetas inmaduras.")
        partitions[name] = subset
        details[name] = {
            "rows": len(subset), "no_show": int(subset.target.sum()),
            "purged_immature_labels": int((mask & ~retained).sum()),
            "prediction_min": subset.prediction_at.min().isoformat(),
            "prediction_max": subset.prediction_at.max().isoformat(),
            "appointment_min": subset.appointment_at.min().isoformat(),
            "appointment_max": subset.appointment_at.max().isoformat(),
            "max_assumed_label_available_at": known.loc[retained].max().isoformat(),
            "label_cutoff_exclusive": cutoffs[name].isoformat(),
        }
    client_sets = {name: set(part.client_id) for name, part in partitions.items()}
    metadata = {
        "split_by": "prediction_at", "validation_start": val.isoformat(), "test_start": test.isoformat(),
        "test_end": end.isoformat(), "labels_observed_until": observed.isoformat(),
        "label_delay_hours": label_delay_hours,
        "label_availability_policy": "appointment_at + externally verified maximum delay; actual availability not in CSV",
        "partitions": details, "outside_booking_windows": int((frame.prediction_at >= end).sum()),
        "client_overlap_train_test": len(client_sets["train"] & client_sets["test"]),
        "client_overlap_train_validation": len(client_sets["train"] & client_sets["validation"]),
        "scope": "Reservas de clientes nuevos y recurrentes; no estima generalización exclusiva a clientes nuevos.",
    }
    return TemporalSplit(**partitions, metadata=metadata)
