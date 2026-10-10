"""Fixture pública artificial para controles de startup, nunca una fila del CSV."""


def synthetic_features():
    return {
        "age_at_booking": 28, "client_sex": "F", "booking_lead_days": 9.5,
        "appointment_month": 11, "appointment_weekday": 2, "appointment_hour": 15,
        "duration_minutes": 40, "is_fwa": False, "clinic_id": 1,
        "scheduled_service_lines": 1, "distinct_body_areas": 1, "single_body_area_id": 101,
        "has_medical_evaluation": False, "has_type4_service": False,
        "previous_attended": 3, "previous_no_show": 1, "previous_no_show_rate": .25,
        "days_since_previous_attended": 30, "days_since_previous_no_show": 110,
    }
