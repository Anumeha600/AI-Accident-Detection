"""
PHASE 5 -- Real-time pipeline glue.
--------------------------------------
Small, UI-independent pieces shared by app.py: a rolling sensor-reading
buffer, the severity lookup, the event-status state machine, and a
simulated GPS random walk. Kept separate from app.py so the logic is
testable and reusable if the UI layer is ever swapped out.

SEVERITY LOGIC (IMPORTANT -- research-integrity requirement):
    The Phase 4 ML model (accident_classifier_model_expanded.joblib)
    predicts an EVENT CLASS only -- one of the six scenario labels. It
    does NOT predict a severity level. The NORMAL/LOW/MEDIUM/HIGH
    severity shown in the dashboard comes from the fixed lookup table
    SEVERITY_BY_LABEL below. This is a hand-authored design choice for
    this demo, not a value learned or validated by the model.
"""

from collections import deque

import numpy as np

from window_features import WINDOW_SIZE

# --- Severity: a fixed, documented mapping from predicted class -> severity.
# NOT produced by the ML model -- see module docstring above.
SEVERITY_BY_LABEL = {
    "Normal Driving": "NORMAL",
    "Pothole": "LOW",
    "Sharp Turn": "LOW",
    "Hard Braking": "MEDIUM",
    "Minor Accident": "MEDIUM",
    "Severe Accident": "HIGH",
}

# Only this severity level starts an emergency-alert countdown.
ALERT_TRIGGER_SEVERITY = "HIGH"

# These two statuses are "sticky": once reached they stay on screen
# (so the user can actually see them) until Reset/Stop or a fresh
# HIGH-severity event starts a new alert cycle.
STICKY_STATUSES = {"ALERT SENT", "ALERT CANCELLED"}


def severity_for_label(label: str) -> str:
    """Rule-based severity lookup -- see module docstring."""
    return SEVERITY_BY_LABEL.get(label, "NORMAL")


def next_status(prev_status: str, severity: str) -> str:
    """
    Pure state-transition function for the Event Status shown in the UI.
    Purely rule-based (not ML-derived). Rules:

      - Once in "ALERT COUNTDOWN", stay there -- app.py handles the
        countdown/cancel/timeout transitions out of this state.
      - A new HIGH-severity prediction (re-)starts an alert countdown,
        even overriding a previous ALERT SENT/CANCELLED from earlier in
        the same run (a new severe event deserves a new alert cycle).
      - ALERT SENT / ALERT CANCELLED are sticky (see STICKY_STATUSES).
      - MEDIUM severity shows EVENT DETECTED (no countdown).
      - Otherwise, MONITORING.
    """
    if prev_status == "ALERT COUNTDOWN":
        return "ALERT COUNTDOWN"
    if severity == ALERT_TRIGGER_SEVERITY:
        return "ALERT COUNTDOWN"
    if prev_status in STICKY_STATUSES:
        return prev_status
    if severity == "MEDIUM":
        return "EVENT DETECTED"
    return "MONITORING"


class RollingWindowBuffer:
    """Keeps the most recent WINDOW_SIZE sensor readings (a sliding window)."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window_size = window_size
        self._buffer = deque(maxlen=window_size)

    def push(self, reading: dict):
        self._buffer.append(reading)

    def is_full(self) -> bool:
        return len(self._buffer) == self._window_size

    def as_list(self):
        return list(self._buffer)

    def reset(self):
        self._buffer.clear()


# Arbitrary simulated starting point -- not tied to any real location.
GPS_ORIGIN = (12.9716, 77.5946)


def simulate_gps_step(rng, previous_lat, previous_lon, step_deg: float = 0.00015):
    """
    SIMULATED GPS: one step of a small random walk around the previous
    point. This is NOT a real GPS reading -- always shown labeled
    'GPS: SIMULATED' in the UI.
    """
    lat = previous_lat + rng.uniform(-step_deg, step_deg)
    lon = previous_lon + rng.uniform(-step_deg, step_deg)
    return lat, lon


def accel_magnitude(reading: dict) -> float:
    return float(np.sqrt(
        reading["accel_x_g"] ** 2 + reading["accel_y_g"] ** 2 + reading["accel_z_g"] ** 2
    ))


def gyro_magnitude(reading: dict) -> float:
    return float(np.sqrt(
        reading["gyro_x_dps"] ** 2 + reading["gyro_y_dps"] ** 2 + reading["gyro_z_dps"] ** 2
    ))
