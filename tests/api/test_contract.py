import pytest

from depi_ml.api.schemas.features import PredictionFeatures
from depi_ml.datasets.schema import PREDICTOR_COLUMNS


def test_exact_canonical_features():
    assert list(PredictionFeatures.model_fields) == PREDICTOR_COLUMNS


@pytest.mark.parametrize("column", PREDICTOR_COLUMNS)
def test_all_features_required_even_nullable(client, headers, payload, column):
    del payload["features"][column]
    r = client.post("/internal/v1/predictions", headers=headers, json=payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_FEATURES"


@pytest.mark.parametrize("name", ["target", "outcome", "client_id", "appointment_id", "label_recorded_at", "voucher_policy", "no_show_uid_minus_one"])
def test_forbidden_features_no_input_leak(client, headers, payload, name):
    payload["features"][name] = "sensitive-value"
    r = client.post("/internal/v1/predictions/compare", headers=headers, json=payload)
    assert r.status_code == 422
    assert "sensitive-value" not in r.text


@pytest.mark.parametrize("field,value", [
    ("age_at_booking", 15), ("age_at_booking", 96), ("age_at_booking", 28.0),
    ("client_sex", "other"), ("booking_lead_days", 0), ("booking_lead_days", -1),
    ("booking_lead_days", "9.5"), ("booking_lead_days", True), ("booking_lead_days", {}),
    ("appointment_month", 0), ("appointment_month", 13), ("appointment_weekday", 0),
    ("appointment_weekday", 8), ("appointment_hour", -1), ("appointment_hour", 24),
    ("is_fwa", "f"), ("is_fwa", 0), ("clinic_id", 0), ("clinic_id", True),
    ("scheduled_service_lines", -1), ("distinct_body_areas", 2), ("single_body_area_id", None),
    ("previous_attended", -1), ("previous_no_show_rate", .7), ("previous_no_show_rate", None),
    ("days_since_previous_attended", -1), ("days_since_previous_no_show", -1),
    ("booking_lead_days", None), ("duration_minutes", -1),
])
def test_invalid_feature_values(client, headers, payload, field, value):
    payload["features"][field] = value
    assert client.post("/internal/v1/predictions", headers=headers, json=payload).status_code == 422


@pytest.mark.parametrize("nullable", ["age_at_booking", "client_sex", "duration_minutes", "is_fwa", "has_medical_evaluation", "has_type4_service", "days_since_previous_attended", "days_since_previous_no_show"])
def test_nullable_fields_preserved(client, registry, headers, payload, nullable):
    payload["features"][nullable] = None
    assert client.post("/internal/v1/predictions", headers=headers, json=payload).status_code == 200
    assert registry.get("random_forest_full_v1").calls[0][nullable] is None


def test_no_history_no_area_and_rounding(client, headers, payload):
    payload["features"].update(previous_attended=0, previous_no_show=0, previous_no_show_rate=None,
                               days_since_previous_attended=None, days_since_previous_no_show=None,
                               distinct_body_areas=0, single_body_area_id=None)
    assert client.post("/internal/v1/predictions", headers=headers, json=payload).status_code == 200
    payload["features"].update(previous_attended=2, previous_no_show=1, previous_no_show_rate=.333333)
    assert client.post("/internal/v1/predictions", headers=headers, json=payload).status_code == 200


@pytest.mark.parametrize("patch", [dict(previous_attended=0, days_since_previous_attended=1,
                previous_no_show_rate=1.), dict(previous_no_show=0, days_since_previous_no_show=1,
                previous_no_show_rate=0.), dict(previous_attended=0, previous_no_show=0,
                previous_no_show_rate=0.), dict(distinct_body_areas=0, single_body_area_id=101)])
def test_cross_field_incoherence(client, headers, payload, patch):
    payload["features"].update(patch)
    assert client.post("/internal/v1/predictions", headers=headers, json=payload).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json(client, headers, payload, token):
    import json
    raw = json.dumps(payload).replace('"booking_lead_days": 9.5', '"booking_lead_days": ' + token)
    r = client.post("/internal/v1/predictions", headers={**headers, "Content-Type": "application/json"}, content=raw)
    assert r.status_code == 422


def test_snapshot_and_extra_metadata_rejected(client, headers, payload):
    payload["feature_context"]["snapshot_type"] = "24h_before"
    assert client.post("/internal/v1/predictions", headers=headers, json=payload).status_code == 422
    payload["feature_context"]["snapshot_type"] = "booking_creation"
    payload["client_id"] = 123
    assert client.post("/internal/v1/predictions", headers=headers, json=payload).status_code == 422
