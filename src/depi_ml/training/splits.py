"""Cortes por prediction_at usando marcas registradas por cita, aún bajo revisión."""

from dataclasses import dataclass

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


def temporal_split(frame, validation_start, test_start, test_end, labels_observed_until, *, require_both_classes=True):
    val, test, end, observed = map(utc_timestamp, [validation_start, test_start, test_end, labels_observed_until])
    if not val < test < end:
        raise Phase2Error("Se requiere validation_start < test_start < test_end.")
    if frame[["prediction_at", "appointment_at", "target"]].isna().any().any() or not frame.target.isin([0, 1]).all():
        raise Phase2Error("Fechas o etiquetas inválidas para dividir temporalmente.")
    if (frame.prediction_at >= frame.appointment_at).any():
        raise Phase2Error("Reserva posterior o simultánea a la cita.")
    if "label_recorded_at" not in frame or not isinstance(frame.label_recorded_at.dtype, pd.DatetimeTZDtype):
        raise Phase2Error("Falta label_recorded_at con zona horaria del auxiliar validado; no se admite demora global.")
    known = frame.label_recorded_at
    if (known < frame.appointment_at).any():
        raise Phase2Error("Fechas inconsistentes: etiqueta registrada antes de la cita.")
    masks = {
        "train": frame.prediction_at < val,
        "validation": (frame.prediction_at >= val) & (frame.prediction_at < test),
        "test": (frame.prediction_at >= test) & (frame.prediction_at < end),
    }
    cutoffs = {"train": val, "validation": test, "test": observed}
    partitions, details = {}, {}
    for name, mask in masks.items():
        retained = mask & (known < cutoffs[name])
        subset = frame.loc[retained].sort_values("prediction_at", kind="stable").copy()
        if require_both_classes and (subset.empty or subset.target.nunique() != 2):
            raise Phase2Error(f"Partición {name} vacía o con una sola clase tras purgar etiquetas inmaduras.")
        partitions[name] = subset
        missing = mask & known.isna()
        late = mask & known.notna() & (known >= cutoffs[name])
        details[name] = {
            "bookings_in_window": int(mask.sum()), "rows": len(subset), "no_show": int(subset.target.sum()),
            "excluded_labels_not_available": int((missing | late).sum()),
            "excluded_missing_label_recorded_at": int(missing.sum()),
            "excluded_recorded_at_or_after_cutoff": int(late.sum()),
            "excluded_by_target": {str(value): {"missing": int((missing & (frame.target == value)).sum()),
               "at_or_after_cutoff": int((late & (frame.target == value)).sum())} for value in [0, 1]},
            "prediction_min": subset.prediction_at.min().isoformat() if len(subset) else None,
            "prediction_max": subset.prediction_at.max().isoformat() if len(subset) else None,
            "appointment_min": subset.appointment_at.min().isoformat() if len(subset) else None,
            "appointment_max": subset.appointment_at.max().isoformat() if len(subset) else None,
            "max_label_recorded_at": known.loc[retained].max().isoformat() if len(subset) else None,
            "label_cutoff_exclusive": cutoffs[name].isoformat(),
        }
    client_sets = {name: set(part.client_id) for name, part in partitions.items()}
    metadata = {
        "split_by": "prediction_at", "validation_start": val.isoformat(), "test_start": test.isoformat(),
        "test_end": end.isoformat(), "labels_observed_until": observed.isoformat(),
        "label_availability_policy": "label_recorded_at strictly before cutoff; recorded timestamp proxy subject to human validation",
        "partitions": details, "outside_booking_windows": int((frame.prediction_at >= end).sum()),
        "client_overlap_train_test": len(client_sets["train"] & client_sets["test"]),
        "client_overlap_train_validation": len(client_sets["train"] & client_sets["validation"]),
        "scope": "Reservas de clientes nuevos y recurrentes; no estima generalización exclusiva a clientes nuevos.",
    }
    return TemporalSplit(**partitions, metadata=metadata)
