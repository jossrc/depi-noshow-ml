"""Controles observables en el CSV y límites de la evidencia retrospectiva."""

import numpy as np
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from depi_ml.analysis.dataset import LocalDataset, Phase2Error
from depi_ml.datasets.schema import PREDICTOR_COLUMNS


REVIEW_REQUIREMENTS = {
    "history_point_in_time": "Revisar SQL completo: known_at < prediction_at, sin fila propia ni eventos futuros; auditar event_at y disponibilidad real.",
    "booking_snapshot": "Acreditar que horario, clínica, duración, sexo y líneas SCHEDULED reflejan la reserva y no modificaciones posteriores, servicios realizados o facturados.",
    "label_availability": "Validar humanamente label_recorded_at del auxiliar: LaserGCloseDate en America/Lima para asistencias y outcome_resolved_at para no-shows. Son marcas registradas, no disponibilidad real certificada. Revisar cierres tardíos, fechas faltantes, modificaciones/ingesta y sesgo de exclusión por corte; no usar una demora global.",
    "catalog_snapshot": "Acreditar catálogo histórico/as-of para áreas, tipos y evaluación médica; updated_at actual no demuestra vigencia histórica.",
    "label_definition": "Validar candidatos no-show, conflictos y tracking UID -1, zona horaria y reglas de cierre; no son inasistencias confirmadas por el CSV.",
    "selection_bias": "Documentar exclusión de cancelaciones anticipadas y otras exclusiones: evaluación condicionada a población elegible, no a todas las reservas futuras.",
    "uncertain_semantics": "Documentar is_fwa y convención de weekday; mantener sensibilidad excluyendo is_fwa y variables del catálogo.",
    "temporal_label_stability": "Revisar caídas/cambios temporales por clínica, cobertura, tracking, madurez y versiones de reglas; distinguir cambio operativo real de etiquetas incompletas o reglas distintas.",
}
VERIFIED_REQUIREMENTS = {
    "history_point_in_time", "booking_snapshot", "label_availability", "catalog_snapshot", "label_definition", "temporal_label_stability",
}


def audit_dataset(dataset: LocalDataset, timezone: str = "America/Lima") -> dict:
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise Phase2Error("Zona horaria IANA inválida para el análisis.") from None
    df = dataset.frame
    blockers, warnings, checks = [], [], {}

    def check(name, mask, critical=True):
        count = int(mask.fillna(False).sum())
        checks[name] = count
        if count:
            (blockers if critical else warnings).append(f"{name}: {count} registros; revisar sin corregir automáticamente.")

    check("target_invalid", ~df.target.isin([0, 1]))
    check("missing_identifiers", df[["appointment_id", "client_id"]].isna().any(axis=1))
    check("duplicate_appointments", df.appointment_id.duplicated(keep=False))
    check("missing_dates", df[["prediction_at", "appointment_at"]].isna().any(axis=1))
    check("prediction_not_before_appointment", df.prediction_at >= df.appointment_at)
    check("missing_audit_or_version", df[["no_show_uid_minus_one", "feature_version", "built_at"]].isna().any(axis=1))
    check("uid_flag_on_attendance", (df.no_show_uid_minus_one == 1) & (df.target != 1))
    nonnegative = ["booking_lead_days", "duration_minutes", "scheduled_service_lines", "distinct_body_areas",
                   "previous_attended", "previous_no_show", "days_since_previous_attended", "days_since_previous_no_show"]
    for column in nonnegative:
        check(f"negative_{column}", df[column] < 0)
    check("missing_history_counts", df[["previous_attended", "previous_no_show"]].isna().any(axis=1))
    total = df.previous_attended + df.previous_no_show
    expected = df.previous_no_show.div(total.where(total > 0))
    check("history_rate_inconsistent", ((total > 0) & (df.previous_no_show_rate.isna() |
          ~np.isclose(df.previous_no_show_rate, expected, atol=1e-6, rtol=1e-6))) |
          ((total == 0) & df.previous_no_show_rate.notna() & (df.previous_no_show_rate != 0)))
    for event in ["attended", "no_show"]:
        check(f"recency_without_{event}_history", (df[f"previous_{event}"] == 0) & df[f"days_since_previous_{event}"].notna())
        check(f"missing_{event}_recency", (df[f"previous_{event}"] > 0) & df[f"days_since_previous_{event}"].isna())
    actual_lead = (df.appointment_at - df.prediction_at).dt.total_seconds() / 86400
    check("lead_date_disagreement", df.booking_lead_days.notna() & ~np.isclose(df.booking_lead_days, actual_lead, atol=1e-5), False)
    check("age_outside_0_120", (df.age_at_booking < 0) | (df.age_at_booking > 120), False)
    check("zero_duration", df.duration_minutes == 0, False)
    check("areas_exceed_lines", df.distinct_body_areas > df.scheduled_service_lines, False)
    check("single_area_id_inconsistent", df.single_body_area_id.notna() & (df.distinct_body_areas != 1), False)
    for column, low, high in [("appointment_month", 1, 12), ("appointment_weekday", 0, 6), ("appointment_hour", 0, 23)]:
        check(f"range_{column}", df[column].notna() & ~df[column].between(low, high))
    local = df.appointment_at.dt.tz_convert(timezone)
    check("month_calendar_disagreement", df.appointment_month.notna() & (df.appointment_month != local.dt.month), False)
    check("hour_calendar_disagreement", df.appointment_hour.notna() & (df.appointment_hour != local.dt.hour), False)
    check("weekday_calendar_disagreement", df.appointment_weekday.notna() & (df.appointment_weekday != (local.dt.dayofweek + 1) % 7), False)

    # Compatibilidad observable de historial vs. eventos dentro del CSV. No demuestra known_at.
    # Cuenta por cliente eventos estrictamente anteriores a la reserva, sin joins cuadráticos.
    comparable, exceeding = 0, 0
    clients = np.unique(df.client_id.to_numpy(), return_inverse=True)[1]
    record_type = np.dtype([("client", "i8"), ("time", "i8")])
    probes = np.empty(len(df), dtype=record_type)
    probes["client"], probes["time"] = clients, df.prediction_at.astype("int64").to_numpy()
    starts = probes.copy()
    starts["time"] = np.iinfo(np.int64).min
    for value, column in [(0, "previous_attended"), (1, "previous_no_show")]:
        mask = df.target.to_numpy() == value
        events = np.empty(int(mask.sum()), dtype=record_type)
        events["client"], events["time"] = clients[mask], df.appointment_at.astype("int64").to_numpy()[mask]
        events.sort(order=["client", "time"])
        observed = np.searchsorted(events, probes, side="left") - np.searchsorted(events, starts, side="left")
        comparable += int((observed > 0).sum())
        exceeding += int((df[column].to_numpy() > observed).sum())
    checks["history_comparable_client_events"] = comparable
    checks["history_count_exceeds_observed_csv_events"] = exceeding
    warnings.extend([
        "Historial: eventos fuera del CSV y known_at ausente impiden reconstrucción exacta. Exceder eventos observados no prueba fuga; fechas ordenadas tampoco la descartan.",
        "ETL completo de construcción no acreditado por este análisis; bds/analytics.sql contiene DDL y una vista de clasificación, no la consulta de las 19 features.",
        "El CSV no incluye fechas registradas de etiquetas. El auxiliar permite cortes por cita, pero esas marcas requieren revisión humana como aproximación de disponibilidad real; los snapshots de reserva/catálogo también siguen pendientes.",
        "UID -1 se compara como subgrupo de auditoría; diferencias descriptivas no validan la etiqueta ni identifican un efecto causal.",
        "Cambios mensuales pueden reflejar mezcla de clínicas, estacionalidad, meses incompletos o reglas; no prueban concept drift.",
    ])
    monthly = df.assign(period=local.dt.strftime("%Y-%m")).groupby("period").target.agg(["size", "mean"])
    previous_rate, previous_size = monthly["mean"].shift(), monthly["size"].shift()
    se = np.sqrt(monthly["mean"] * (1 - monthly["mean"]) / monthly["size"] + previous_rate * (1 - previous_rate) / previous_size)
    flags = (monthly["size"] >= 100) & (previous_size >= 100) & ((monthly["mean"] - previous_rate).abs() >= .05) & ((monthly["mean"] - previous_rate).abs() > 5 * se)
    changes = [{"month": period, "rows": int(monthly.loc[period, "size"]),
                "no_show_rate": float(monthly.loc[period, "mean"]), "previous_rate": float(previous_rate.loc[period])}
               for period in monthly.index[flags]]
    if changes:
        warnings.append("Cambios mensuales grandes detectados (>=5 puntos porcentuales y >5 errores estándar nominales): requieren revisión de etiquetas/cobertura; no constituyen una prueba formal de drift.")
    identical = []
    for i, first in enumerate(PREDICTOR_COLUMNS):
        for second in PREDICTOR_COLUMNS[i + 1:]:
            if df[first].equals(df[second]):
                identical.append([first, second])
    if identical:
        warnings.append("Existen predictoras idénticas en el CSV; revisar redundancia e interpretar con cautela permutation importance y SHAP.")
    return {
        "dataset_sha256": dataset.sha256, "feature_versions": dataset.manifest["feature_versions"],
        "rows": len(df), "timezone": timezone, "predictors": list(PREDICTOR_COLUMNS),
        "csv_checks": checks, "hard_blockers": blockers, "warnings": warnings,
        "training_status": "blocked_pending_methodological_review",
        "pending_review": REVIEW_REQUIREMENTS,
        "label_available_at_in_csv": False,
        "temporal_rate_alerts": changes,
        "identical_predictor_pairs": identical,
        "conclusion": "Integridad y coherencia observables no certifican ausencia de leakage ni validez científica.",
    }
