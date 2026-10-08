"""realtime_pipeline: next_status() state machine, severity lookup, buffer, helpers."""

import numpy as np
import pytest

from realtime_pipeline import (
    ALERT_TRIGGER_SEVERITY, SEVERITY_BY_LABEL, STICKY_STATUSES, GPS_ORIGIN,
    RollingWindowBuffer, accel_magnitude, gyro_magnitude, next_status,
    severity_for_label, simulate_gps_step,
)
from window_features import WINDOW_SIZE

ALL_SEVERITIES = ["NORMAL", "LOW", "MEDIUM", "HIGH"]
ALL_STATUSES = ["MONITORING", "EVENT DETECTED", "ALERT COUNTDOWN", "ALERT SENT", "ALERT CANCELLED"]


# --- severity ------------------------------------------------------------

def test_severity_lookup_table():
    assert SEVERITY_BY_LABEL == {
        "Normal Driving": "NORMAL", "Pothole": "LOW", "Sharp Turn": "LOW",
        "Hard Braking": "MEDIUM", "Minor Accident": "MEDIUM", "Severe Accident": "HIGH",
    }


def test_only_severe_accident_triggers_alert():
    assert ALERT_TRIGGER_SEVERITY == "HIGH"
    assert [k for k, v in SEVERITY_BY_LABEL.items() if v == "HIGH"] == ["Severe Accident"]


def test_unknown_label_defaults_to_normal():
    assert severity_for_label("Something Else") == "NORMAL"


# --- next_status: transitions -----------------------------------------------

def test_monitoring_stays_monitoring_for_normal_and_low():
    for sev in ("NORMAL", "LOW"):
        assert next_status("MONITORING", sev) == "MONITORING"


def test_medium_shows_event_detected_without_countdown():
    assert next_status("MONITORING", "MEDIUM") == "EVENT DETECTED"


def test_event_detected_returns_to_monitoring_when_calm():
    assert next_status("EVENT DETECTED", "NORMAL") == "MONITORING"
    assert next_status("EVENT DETECTED", "LOW") == "MONITORING"


def test_high_starts_countdown_from_any_status():
    for prev in ALL_STATUSES:
        assert next_status(prev, "HIGH") == "ALERT COUNTDOWN"


def test_countdown_is_held_by_next_status_for_every_severity():
    # app.py owns the countdown -> SENT/CANCELLED transitions; the pure
    # function must never leave the countdown on its own.
    for sev in ALL_SEVERITIES:
        assert next_status("ALERT COUNTDOWN", sev) == "ALERT COUNTDOWN"


def test_sent_and_cancelled_are_sticky_below_high():
    assert STICKY_STATUSES == {"ALERT SENT", "ALERT CANCELLED"}
    for prev in STICKY_STATUSES:
        for sev in ("NORMAL", "LOW", "MEDIUM"):
            assert next_status(prev, sev) == prev


def test_new_high_event_overrides_sticky_status():
    assert next_status("ALERT SENT", "HIGH") == "ALERT COUNTDOWN"
    assert next_status("ALERT CANCELLED", "HIGH") == "ALERT COUNTDOWN"


def test_full_table_is_total_and_returns_known_statuses():
    for prev in ALL_STATUSES:
        for sev in ALL_SEVERITIES:
            assert next_status(prev, sev) in ALL_STATUSES


def test_realistic_sequence_through_a_severe_accident():
    status = "MONITORING"
    for label, expected in [
        ("Normal Driving", "MONITORING"),
        ("Hard Braking", "EVENT DETECTED"),
        ("Severe Accident", "ALERT COUNTDOWN"),
        ("Normal Driving", "ALERT COUNTDOWN"),   # held until app.py resolves it
    ]:
        status = next_status(status, severity_for_label(label))
        assert status == expected


# app.py computes remaining = countdown_total - int(elapsed); that logic is
# inline in the Streamlit script (not importable), so this only pins the
# formula's intended behaviour. The app-level behaviour is covered in
# test_app_smoke.py.
@pytest.mark.parametrize("elapsed,remaining", [(0.0, 10), (0.9, 10), (1.0, 9), (9.99, 1), (10.0, 0), (12, -2)])
def test_countdown_formula(elapsed, remaining):
    assert 10 - int(elapsed) == remaining


# --- rolling buffer -----------------------------------------------------------

def test_buffer_fills_then_slides():
    b = RollingWindowBuffer()
    assert not b.is_full()
    for i in range(WINDOW_SIZE - 1):
        b.push({"i": i})
    assert not b.is_full()
    b.push({"i": WINDOW_SIZE - 1})
    assert b.is_full()
    b.push({"i": WINDOW_SIZE})
    assert b.is_full()
    assert [r["i"] for r in b.as_list()] == list(range(1, WINDOW_SIZE + 1))


def test_buffer_reset_empties_it():
    b = RollingWindowBuffer()
    for i in range(WINDOW_SIZE):
        b.push({"i": i})
    b.reset()
    assert not b.is_full() and b.as_list() == []


def test_as_list_returns_a_copy():
    b = RollingWindowBuffer()
    b.push({"i": 0})
    b.as_list().clear()
    assert len(b.as_list()) == 1


# --- magnitudes and GPS ----------------------------------------------------------

def test_magnitudes():
    r = {"accel_x_g": 3, "accel_y_g": 4, "accel_z_g": 0, "gyro_x_dps": 0, "gyro_y_dps": 6, "gyro_z_dps": 8}
    assert accel_magnitude(r) == pytest.approx(5.0)
    assert gyro_magnitude(r) == pytest.approx(10.0)


def test_gps_step_is_a_small_bounded_walk():
    rng = np.random.default_rng(0)
    lat, lon = GPS_ORIGIN
    for _ in range(100):
        nlat, nlon = simulate_gps_step(rng, lat, lon)
        assert abs(nlat - lat) <= 0.00015 and abs(nlon - lon) <= 0.00015
        lat, lon = nlat, nlon
