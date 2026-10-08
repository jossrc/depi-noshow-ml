"""Tablas y gráficos agregados; nunca exporta filas ni identificadores personales."""

from pathlib import Path

import numpy as np
import pandas as pd

from depi_ml.analysis.audit import audit_dataset
from depi_ml.analysis.dataset import Phase2Error, load_dataset, new_output, write_json
from depi_ml.datasets.schema import PREDICTOR_COLUMNS

DESCRIPTIVE_CATEGORIES = ["client_sex", "clinic_id", "single_body_area_id", "is_fwa", "has_medical_evaluation",
                          "has_type4_service", "appointment_month", "appointment_weekday", "appointment_hour"]


def plotting():
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def aggregate(frame, group, min_group=10):
    table = frame.groupby(group, dropna=False).target.agg(rows="size", no_show="sum").reset_index()
    table["attended"] = table.rows - table.no_show
    table["no_show_rate"] = table.no_show / table.rows
    # Suprimir celdas pequeñas y sus etiquetas; no publicar registros individuales.
    return table.loc[table.rows >= min_group].copy()


def numeric_summary(frame, columns):
    records = []
    for column in columns:
        data = frame[column].dropna()
        stats = {"variable": column, "non_null": len(data)}
        for label, q in [("min", 0), ("p01", .01), ("p25", .25), ("median", .5), ("p75", .75), ("p99", .99), ("max", 1)]:
            stats[label] = float(data.quantile(q)) if len(data) else None
        stats["mean"] = float(data.mean()) if len(data) else None
        records.append(stats)
    return records


def analyze(csv_path: Path, manifest_path: Path | None, output: Path, timezone="America/Lima", min_group=10) -> Path:
    if min_group < 10:
        raise Phase2Error("El mínimo para grupos publicados es 10.")
    dataset = load_dataset(csv_path, manifest_path)
    report = audit_dataset(dataset, timezone)
    out = new_output(output, dataset)
    df = dataset.frame.copy()
    df["period"] = df.appointment_at.dt.tz_convert(timezone).dt.strftime("%Y-%m")
    report["suppression_minimum"] = min_group
    missing = pd.DataFrame({"variable": df.columns, "null_count": df.isna().sum().values})
    missing["null_percent"] = missing.null_count / len(df) * 100
    missing.to_csv(out / "missing_values.csv", index=False)
    numeric = [c for c in PREDICTOR_COLUMNS if c not in DESCRIPTIVE_CATEGORIES]
    pd.DataFrame(numeric_summary(df, numeric)).to_csv(out / "numeric_summary.csv", index=False)
    df[numeric].corr().to_csv(out / "numeric_correlations.csv")
    monthly = aggregate(df, "period", min_group)
    monthly["rate_change_previous_month"] = monthly.no_show_rate.diff()
    monthly["standard_error"] = np.sqrt(monthly.no_show_rate * (1 - monthly.no_show_rate) / monthly.rows)
    monthly.to_csv(out / "monthly.csv", index=False)
    aggregate(df, "clinic_id", min_group).to_csv(out / "clinics.csv", index=False)
    aggregate(df, ["period", "clinic_id"], min_group).to_csv(out / "clinic_month.csv", index=False)
    aggregate(df, "target", min_group).to_csv(out / "classes.csv", index=False)
    uid = aggregate(df, "no_show_uid_minus_one", min_group)
    uid.to_csv(out / "uid_groups.csv", index=False)
    cohorts = []
    for name, subset in [("system_tracking", df[df.no_show_uid_minus_one == 1]),
                         ("other_no_show", df[(df.no_show_uid_minus_one == 0) & (df.target == 1)]),
                         ("other_tracking", df[df.no_show_uid_minus_one == 0])]:
        if len(subset) >= min_group:
            for row in numeric_summary(subset, numeric):
                cohorts.append({"cohort": name, **row})
    pd.DataFrame(cohorts, columns=["cohort", "variable", "non_null", "min", "p01", "p25", "median", "p75", "p99", "max", "mean"]).to_csv(out / "uid_feature_comparison.csv", index=False)
    aggregate(df, ["period", "no_show_uid_minus_one"], min_group).to_csv(out / "uid_month.csv", index=False)
    candidate_cohorts = df[df.target == 1].copy()
    candidate_cohorts["cohort"] = np.where(candidate_cohorts.no_show_uid_minus_one == 1, "system_tracking_no_show", "other_no_show")
    for column in DESCRIPTIVE_CATEGORIES + ["scheduled_service_lines", "distinct_body_areas"]:
        aggregate(df, column, min_group).to_csv(out / f"distribution_{column}.csv", index=False)
        aggregate(candidate_cohorts, ["cohort", column], min_group).to_csv(out / f"uid_categories_{column}.csv", index=False)
    history_cohorts = df.assign(history_group=np.where(df.previous_attended + df.previous_no_show == 0, "no_prior_history", "prior_history"))
    aggregate(history_cohorts, "history_group", min_group).to_csv(out / "history_groups.csv", index=False)
    for column in ["days_since_previous_attended", "days_since_previous_no_show"]:
        recency = df.assign(recency_group=pd.cut(df[column], [0, 1, 7, 30, 90, float("inf")],
                            labels=["0_to_1_days", "1_to_7_days", "7_to_30_days", "30_to_90_days", "over_90_days"], include_lowest=True).astype(object))
        aggregate(recency, "recency_group", min_group).to_csv(out / f"recency_{column}.csv", index=False)

    special = df[df.client_id == 99467]
    special_report = {"present": bool(len(special)), "detailed_statistics_suppressed": len(special) < min_group,
                      "rows": len(special), "no_show": int(special.target.sum()),
                      "no_show_rate": float(special.target.mean()) if len(special) else None,
                      "uid_flag_count": int(special.no_show_uid_minus_one.sum()),
                      "description": "Cliente atípico solicitado: solo resumen agregado, sin identificador."}
    if len(special) >= min_group:
        special_report.update(rows=len(special), no_show=int(special.target.sum()),
                              no_show_rate=float(special.target.mean()),
                              uid_flag_count=int(special.no_show_uid_minus_one.sum()),
                              summary=numeric_summary(special, ["booking_lead_days", "duration_minutes", "previous_attended", "previous_no_show"]))
        aggregate(special, "period", min_group).to_csv(out / "atypical_cohort_monthly.csv", index=False)
    write_json(out / "atypical_cohort.json", special_report)
    # Valores extremos: conteos IQR descriptivos; no eliminan ni recortan observaciones.
    outliers = []
    for column in numeric:
        values = df[column].dropna()
        q1, q3 = values.quantile([.25, .75])
        iqr = q3 - q1
        outliers.append({"variable": column, "below_iqr_fence": int((values < q1 - 1.5 * iqr).sum()),
                         "above_iqr_fence": int((values > q3 + 1.5 * iqr).sum()), "iqr": float(iqr) if len(values) else None})
    write_json(out / "outliers.json", outliers)
    report["classes"] = {"attended": int((df.target == 0).sum()), "no_show": int((df.target == 1).sum()),
                         "no_show_rate": float(df.target.mean())}
    write_json(out / "audit.json", report)
    template = {"dataset_sha256": dataset.sha256, "feature_versions": dataset.manifest["feature_versions"],
                "reviewer": "", "reviewed_at": "", "label_delay_hours": None,
                "checks": {name: {"status": "pending", "evidence": "", "instruction": instruction}
                           for name, instruction in report["pending_review"].items()}}
    write_json(out / "methodology_review_template.json", template)
    lines = ["# Auditoría metodológica", "", f"Filas: {len(df)}. SHA-256: `{dataset.sha256}`.",
             "", "Entrenamiento bloqueado hasta aportar evidencia externa vinculada a este hash.",
             "La plantilla pendiente no autoriza entrenamiento. No se certifica ausencia de fuga.", "",
             "## Controles del CSV", ""]
    lines += [f"- {name}: {count}." for name, count in report["csv_checks"].items()]
    lines += ["", "## Anomalías críticas", ""] + [f"- {item}" for item in (report["hard_blockers"] or ["Sin errores críticos observables; faltan controles externos."])]
    lines += ["", "## Riesgos y revisión SQL pendiente", ""] + [f"- **{name}**: {value}" for name, value in report["pending_review"].items()]
    lines += ["", "## Interpretación", ""] + [f"- {item}" for item in report["warnings"]]
    lines += ["", "## Alertas temporales descriptivas", ""] + [f"- {r['month']}: {r['no_show_rate']:.2%} frente a {r['previous_rate']:.2%} el mes previo ({r['rows']} citas)." for r in report["temporal_rate_alerts"]]
    lines += ["", "Las distribuciones mensuales usan fecha de cita en la zona indicada; los cortes de entrenamiento usan fecha de reserva en UTC.",
              "Los meses inicial/final pueden ser incompletos. Las diferencias de tasas son descriptivas y no tests de causalidad.",
              "Los grupos con menos del mínimo se omiten. Tablas y gráficos no contienen filas personales.",
              "La cohorte atípica se presenta aparte; su exclusión no se realiza automáticamente.",
              "Las tablas de UID comparan también candidatos no-show entre sí, evitando atribuir a UID el efecto de comparar etiquetas distintas."]
    (out / "methodology.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt = plotting()
    for column in ["age_at_booking", "booking_lead_days", "duration_minutes", "previous_attended", "previous_no_show", "days_since_previous_attended", "days_since_previous_no_show"]:
        fig, ax = plt.subplots()
        for target, label in [(0, "Asistencia"), (1, "Candidato no-show")]:
            values = df.loc[df.target == target, column].dropna()
            if len(values) >= min_group:
                ax.hist(values, bins=40, alpha=.5, label=label)
        ax.set(xlabel=column, ylabel="Frecuencia", title="Distribución completa (sin recorte)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"hist_{column}.png")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.period, monthly.no_show_rate, marker="o")
    ax.set(ylabel="Proporción candidato no-show", xlabel=f"Mes ({timezone})")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(out / "monthly_rate.png")
    plt.close(fig)
    write_json(out / "completion.json", {"status": "complete", "dataset_sha256": dataset.sha256,
               "training_status": report["training_status"]})
    return out
