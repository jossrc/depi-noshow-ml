"""Auditoría agregada de PostgreSQL existente: SELECT, sin exportar filas personales.

Ejecutar desde la raíz. El destino debe ser nuevo bajo reports/.
No ejecuta SQL de bds/, ETL, sincronización ni administración de Docker.
"""

import argparse
from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from depi_ml.config import Settings
from depi_ml.datasets.validator import inspect_dataset


# SQL fijo revisable. Sin SQL/identificadores aportados por argumentos o datos.
QUERIES = {
    "calendar": """
        SELECT count(*) AS rows,
          min(appointment_weekday) AS weekday_min, max(appointment_weekday) AS weekday_max,
          count(*) FILTER (WHERE appointment_weekday IS DISTINCT FROM
            extract(isodow FROM appointment_at AT TIME ZONE 'America/Lima')) AS iso_weekday_mismatches,
          count(*) FILTER (WHERE appointment_weekday IS DISTINCT FROM
            extract(dow FROM appointment_at AT TIME ZONE 'America/Lima')) AS dow_weekday_mismatches,
          count(*) FILTER (WHERE appointment_month IS DISTINCT FROM
            extract(month FROM appointment_at AT TIME ZONE 'America/Lima')) AS month_mismatches,
          count(*) FILTER (WHERE appointment_hour IS DISTINCT FROM
            extract(hour FROM appointment_at AT TIME ZONE 'America/Lima')) AS hour_mismatches,
          min(prediction_at) AS prediction_min, max(prediction_at) AS prediction_max
        FROM analytics.appointment_training_dataset_v1
    """,
    "booking_snapshot_evidence": """
        SELECT count(*) AS rows,
          count(*) FILTER (WHERE d.prediction_at IS DISTINCT FROM
            g."DiaryGCreated" AT TIME ZONE 'America/Lima') AS creation_mismatches,
          count(*) FILTER (WHERE g."DiaryGStamp">g."DiaryGCreated") AS modified_headers,
          count(*) FILTER (WHERE g."DiaryGStamp" IS NULL) AS missing_modification_stamp,
          count(*) FILTER (WHERE o.training_eligible IS FALSE) AS current_ineligible_flag,
          count(*) FILTER (WHERE g."DiaryGDelayed"=-1) AS history_headers_in_dataset
        FROM analytics.appointment_training_dataset_v1 d
        LEFT JOIN raw_flow.diary_gen g ON g."DiaryGID"=d.appointment_id
        LEFT JOIN analytics.appointment_outcomes o USING (appointment_id)
    """,
    "history_recorded_time": """
        SELECT h.event_kind, count(*) AS events,
          count(*) FILTER (WHERE h.known_at IS DISTINCT FROM CASE WHEN h.event_kind=0
            THEN l."LaserGCloseDate" AT TIME ZONE 'America/Lima'
            ELSE o.outcome_resolved_at END) AS label_proxy_mismatches,
          count(*) FILTER (WHERE h.known_at>o.appointment_at) AS known_after_appointment,
          count(*) FILTER (WHERE h.updated_at>h.known_at) AS computed_after_known
        FROM analytics.ml_history_cumulative_v1 h
        LEFT JOIN analytics.appointment_outcomes o USING (appointment_id)
        LEFT JOIN raw_flow.laser_gen l ON h.event_kind=0 AND l."LaserGID"=o.diary_laser_id
        GROUP BY h.event_kind ORDER BY h.event_kind LIMIT 2
    """,
    "outcomes": """
        SELECT outcome::text AS outcome, training_eligible, count(*) AS rows,
          count(*) FILTER (WHERE outcome_resolved_at IS NULL) AS missing_resolved_at,
          count(*) FILTER (WHERE tracking_at_source IS NULL) AS missing_tracking_at
        FROM analytics.appointment_outcomes
        GROUP BY outcome, training_eligible ORDER BY outcome, training_eligible LIMIT 100
    """,
    "history_asof": """
        SELECT count(*) AS rows,
          count(*) FILTER (WHERE h.appointment_id IS NULL) AS no_prior_event,
          count(*) FILTER (WHERE h.known_at >= d.prediction_at) AS non_prior_known_at,
          count(*) FILTER (WHERE d.previous_attended IS DISTINCT FROM
            coalesce(h.prior_including_event_attended,0)) AS attended_count_mismatches,
          count(*) FILTER (WHERE d.previous_no_show IS DISTINCT FROM
            coalesce(h.prior_including_event_no_show,0)) AS no_show_count_mismatches,
          count(*) FILTER (WHERE h.appointment_id=d.appointment_id) AS own_event_matches,
          count(*) FILTER (WHERE h.last_attended_at>d.prediction_at
            OR h.last_no_show_at>d.prediction_at) AS future_recency_events
        FROM analytics.appointment_training_dataset_v1 d
        LEFT JOIN LATERAL (
          SELECT appointment_id, known_at, prior_including_event_attended,
            prior_including_event_no_show, last_attended_at, last_no_show_at
          FROM analytics.ml_history_cumulative_v1
          WHERE client_id=d.client_id AND known_at<d.prediction_at
          ORDER BY known_at DESC, appointment_id DESC LIMIT 1
        ) h ON true
    """,
    "raw_versions": """
        SELECT count(*) AS headers,
          count(*) FILTER (WHERE "DiaryGDelayed"=-1) AS delayed_history_mark,
          count(*) FILTER (WHERE "DiaryGDelayedID"<>0) AS nonzero_link,
          count(*) FILTER (WHERE "DiaryGCreated" IS NULL) AS missing_created,
          count(*) FILTER (WHERE "DiaryGStamp" IS NULL) AS missing_stamp,
          count(*) FILTER (WHERE "DiaryGStamp">"DiaryGCreated") AS modified_after_created,
          count(*) FILTER (WHERE "DiaryGNotAttended"=-1) AS cancellation_mark
        FROM raw_flow.diary_gen
    """,
    "linked_versions": """
        SELECT count(*) AS linked_headers,
          count(*) FILTER (WHERE n."DiaryGID" IS NULL) AS missing_link_target,
          count(*) FILTER (WHERE d."DiaryGID"=n."DiaryGID") AS self_links,
          count(*) FILTER (WHERE d."DiaryGClientID"<>n."DiaryGClientID") AS different_client,
          count(*) FILTER (WHERE d."DiaryGDate" IS DISTINCT FROM n."DiaryGDate"
            OR d."DiaryGStart" IS DISTINCT FROM n."DiaryGStart") AS changed_schedule,
          count(*) FILTER (WHERE n."DiaryGCreated"<d."DiaryGCreated") AS target_created_before_source,
          count(*) FILTER (WHERE d."DiaryGDelayed"=-1) AS source_history_mark
        FROM raw_flow.diary_gen d
        LEFT JOIN raw_flow.diary_gen n ON n."DiaryGID"=d."DiaryGDelayedID"
        WHERE d."DiaryGDelayedID"<>0
    """,
    "linked_detail_counts": """
        WITH details AS (
          SELECT "DiaryGID" AS id, count(*) AS lines
          FROM raw_flow.diary_det GROUP BY "DiaryGID"
        )
        SELECT count(*) AS linked_headers,
          count(*) FILTER (WHERE a.id IS NULL) AS source_without_details,
          count(*) FILTER (WHERE n."DiaryGID" IS NOT NULL AND b.id IS NULL) AS target_without_details,
          count(*) FILTER (WHERE a.id IS NOT NULL AND b.id IS NOT NULL) AS both_have_details,
          count(*) FILTER (WHERE a.lines IS DISTINCT FROM b.lines) AS different_detail_counts
        FROM raw_flow.diary_gen d
        LEFT JOIN raw_flow.diary_gen n ON n."DiaryGID"=d."DiaryGDelayedID"
        LEFT JOIN details a ON a.id=d."DiaryGID"
        LEFT JOIN details b ON b.id=n."DiaryGID"
        WHERE d."DiaryGDelayedID"<>0
    """,
    "linked_detail_sample": """
        WITH links AS MATERIALIZED (
          SELECT * FROM raw_flow.diary_gen WHERE "DiaryGDelayedID"<>0
          ORDER BY "DiaryGID" LIMIT 10000
        ), refs AS (
          SELECT "DiaryGID" AS id FROM links UNION SELECT "DiaryGDelayedID" FROM links
        ), details AS (
          SELECT "DiaryGID" AS id, count(*) AS lines,
            array_agg(ROW("DiaryBudgetID","DiaryZoneID","DiaryPackID",
              "DiaryVoucherID","DiarySessionID") ORDER BY "DiaryBudgetID",
              "DiaryZoneID","DiaryPackID","DiaryVoucherID","DiarySessionID") AS signature
          FROM raw_flow.diary_det JOIN refs ON refs.id="DiaryGID" GROUP BY "DiaryGID"
        )
        SELECT count(*) AS linked_headers,
          count(*) FILTER (WHERE n."DiaryGID" IS NULL) AS missing_link_target,
          count(*) FILTER (WHERE d."DiaryGID"=n."DiaryGID") AS self_links,
          count(*) FILTER (WHERE d."DiaryGClientID"<>n."DiaryGClientID") AS different_client,
          count(*) FILTER (WHERE d."DiaryGDate" IS DISTINCT FROM n."DiaryGDate"
            OR d."DiaryGStart" IS DISTINCT FROM n."DiaryGStart") AS changed_schedule,
          count(*) FILTER (WHERE a.id IS NULL) AS source_without_details,
          count(*) FILTER (WHERE n."DiaryGID" IS NOT NULL AND b.id IS NULL) AS target_without_details,
          count(*) FILTER (WHERE a.id IS NOT NULL AND b.id IS NOT NULL) AS both_have_details,
          count(*) FILTER (WHERE a.id IS NOT NULL AND b.id IS NOT NULL
            AND a.signature=b.signature) AS equal_detail_multisets,
          count(*) FILTER (WHERE a.id IS NOT NULL AND b.id IS NOT NULL
            AND a.signature<>b.signature) AS changed_detail_multisets,
          count(*) FILTER (WHERE n."DiaryGCreated"<d."DiaryGCreated") AS target_created_before_source,
          count(*) FILTER (WHERE d."DiaryGDelayed"=-1) AS source_history_mark
        FROM links d
        LEFT JOIN raw_flow.diary_gen n ON n."DiaryGID"=d."DiaryGDelayedID"
        LEFT JOIN details a ON a.id=d."DiaryGID"
        LEFT JOIN details b ON b.id=n."DiaryGID"
        WHERE d."DiaryGDelayedID"<>0
    """,
    "chain_structure": """
        WITH RECURSIVE walk AS (
          SELECT "DiaryGID" AS origin, "DiaryGID" AS node, "DiaryGDelayedID" AS next,
            ARRAY["DiaryGID"] AS path, 0 AS depth, false AS cycle
          FROM raw_flow.diary_gen WHERE "DiaryGDelayedID"<>0
          UNION ALL
          SELECT w.origin, n."DiaryGID", n."DiaryGDelayedID", w.path||n."DiaryGID",
            w.depth+1, n."DiaryGID"=ANY(w.path)
          FROM walk w JOIN raw_flow.diary_gen n ON n."DiaryGID"=w.next
          WHERE w.next<>0 AND w.depth<12 AND NOT w.cycle
        ), terminal AS (
          SELECT DISTINCT ON (origin) node,next,depth,cycle FROM walk
          ORDER BY origin, depth DESC
        )
        SELECT count(*) AS origins,
          count(*) FILTER (WHERE next=0 AND NOT cycle) AS structural_complete,
          count(*) FILTER (WHERE cycle) AS cyclic,
          count(*) FILTER (WHERE depth=12 AND next<>0 AND NOT cycle) AS depth_cap_reached,
          count(*) FILTER (WHERE next<>0 AND depth<12 AND NOT cycle) AS broken,
          max(depth) AS maximum_depth_observed
        FROM terminal
    """,
    "orphan_details": """
        SELECT count(*) AS detail_rows,
          count(*) FILTER (WHERE g."DiaryGID" IS NULL) AS orphan_rows
        FROM raw_flow.diary_det d LEFT JOIN raw_flow.diary_gen g USING ("DiaryGID")
    """,
    "events": """
        SELECT event_type::text AS event_type, count(*) AS rows,
          count(*) FILTER (WHERE event_at IS NULL) AS missing_event_at,
          count(*) FILTER (WHERE computed_at>event_at) AS computed_after_event
        FROM analytics.appointment_events GROUP BY event_type ORDER BY event_type LIMIT 100
    """,
    "service_phases": """
        SELECT service_phase::text AS phase, count(*) AS lines,
          count(DISTINCT appointment_id) AS appointments
        FROM analytics.appointment_services GROUP BY service_phase ORDER BY service_phase LIMIT 10
    """,
    "cancelled_timestamps": """
        SELECT count(*) AS cancelled,
          count(*) FILTER (WHERE outcome_resolved_at IS NULL) AS missing_resolved,
          count(*) FILTER (WHERE tracking_at_source IS NULL) AS missing_tracking,
          count(*) FILTER (WHERE outcome_resolved_at<appointment_at) AS resolved_before_appointment,
          count(*) FILTER (WHERE computed_at>outcome_resolved_at) AS computed_after_resolved
        FROM analytics.appointment_outcomes WHERE outcome::text='CANCELLED'
    """,
}


def json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError("Unsupported aggregate type")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=["dataset_contract", *QUERIES],
                        help="Ejecutar solo categorías fijas sin repetir escaneos previos.")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root / "reports") or output.exists():
        parser.error("Destino nuevo dentro de reports/ requerido.")
    settings = Settings.from_env()
    if (settings.schema, settings.table) != ("analytics", "appointment_training_dataset_v1"):
        parser.error("Origen analítico inesperado; auditoría detenida.")
    report = {"postgres_access_mode": "READ_ONLY", "postgres_write_operations": 0,
              "statement_timeout_ms": 15000, "queries": {},
              "individual_rows_exported": 0}
    try:
        with psycopg.connect(host=settings.host, port=settings.port, dbname=settings.database,
                             user=settings.user, password=settings.password, connect_timeout=5,
                             options="-c default_transaction_read_only=on",
                             application_name="depi-ml-readonly-audit", row_factory=dict_row,
                             autocommit=True) as connection:
            for name in args.only or ("dataset_contract", *QUERIES):
                try:
                    with connection.transaction():
                        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                        connection.execute("SET LOCAL statement_timeout = '15000ms'")
                        identity = connection.execute("""
                          SELECT current_database() AS database, current_user AS role,
                            current_setting('transaction_read_only') AS read_only,
                            current_setting('transaction_isolation') AS isolation
                        """).fetchone()
                        if (identity["database"] != settings.database or identity["role"] != settings.user
                                or identity["read_only"] != "on" or identity["isolation"] != "repeatable read"):
                            raise RuntimeError("Read-only identity guard failed")
                        report["identity"] = identity
                        result = (inspect_dataset(connection, settings) if name == "dataset_contract"
                                  else connection.execute(QUERIES[name]).fetchall())
                        report["queries"][name] = {"status": "PASS", "result": result}
                except psycopg.Error as error:
                    # PostgreSQL messages can contain row values; retain only static type/state.
                    report["queries"][name] = {"status": "BLOCKED", "error_type": type(error).__name__,
                                                "sqlstate": error.sqlstate}
        report["status"] = "PASS" if all(x["status"] == "PASS" for x in report["queries"].values()) else "PARTIAL"
    except psycopg.Error as error:
        report.update(status="BLOCKED", error_type=type(error).__name__, sqlstate=error.sqlstate)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, default=json_default, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "queries": {
        k: v["status"] for k, v in report["queries"].items()}}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
