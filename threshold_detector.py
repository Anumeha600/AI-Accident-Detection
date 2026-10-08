"""
PHASE 8 -- Traditional rule/threshold accident detector (the non-ML baseline).
------------------------------------------------------------------------------
A transparent detector that decides from sensor-derived window features only.
It contains NO machine learning: no model, no probabilities, no scenario
labels or any other ground truth at inference time.

INPUT (identical information to what the Phase 7 ML model receives)
    The window features from window_features.py (computed over one 1-second /
    10-sample window of the 8 raw channels). detect_window() computes them
    from raw readings with the unchanged extract_window_features().

PRIMARY OUTPUT -- binary accident detection
    "ACCIDENT" if ANY of four physically interpretable rules fires, else
    "NORMAL" (meaning "no accident": potholes, hard braking and sharp turns
    are deliberately NOT accidents; see ACCIDENT_SCENARIOS).

        impact_acceleration  accel_mag_max          > T   peak |a| in the window (g)
        angular_rate         gyro_mag_max           > T   peak |omega| (deg/s)
        vibration            vibration_level_mean   > T   mean vibration level
        rapid_deceleration   -(speed_kmh_first_to_last) > T   speed lost across the window (km/h)

    Comparison is STRICT (value > threshold fires; value == threshold does
    not). The result explains itself: triggered_rules + the measured value
    and threshold for each (severity_inputs). No severity is assigned here.

SECONDARY OUTPUT -- 7-way event class (a deliberately limited cascade)
    "event_class" is a best-effort label from a fixed decision cascade:
        accident gate fires ->  Rollover       (roll about x dominates the
                                                other gyro axes by >= 2x)
                                Severe Accident (accel_mag_max >= 6 g)
                                Minor Accident  (otherwise)
        no accident         ->  Sharp Turn      (gyro_z range above noise floor)
                                Hard Braking    (mean longitudinal accel below
                                                 noise floor)
                                Pothole         (vertical accel range above
                                                 noise floor)
                                Normal Driving  (otherwise)
    "Multi-Impact Collision" is NEVER output: a rule set looking at a single
    1-second window cannot tell a second impact from a first one. That is a
    structural limit of the threshold approach, not something hidden.

HOW THRESHOLD VALUES ARE SET (see calibrate_thresholds(); all provenance is
stored in the config and in comparison_report.json)
    * accident-gate thresholds: the 99th percentile of each rule's quantity
      over TRAINING windows of the four NON-accident scenarios. I.e. "above
      what ordinary driving, potholes, hard braking and sharp turns produce
      99% of the time". Recipe fixed before any test evaluation; never tuned.
    * cascade noise floors: 99th percentile (1st for the negative-going
      braking rule) of the quantity over TRAINING Normal Driving windows.
    * rollover_dominance_ratio (2.0), rollover_min_gyro_x_range_dps (100) and
      severe_accel_mag_g (6.0) are AUTHOR-CHOSEN constants from physical
      reasoning (a roll is rotation mostly about one axis; ~6 g is in the
      range commonly associated with severe crash pulses). The author also
      wrote the simulator, so these are not "blind"; they were fixed before the
      test set was evaluated and not changed afterwards.
    The test set is never used. Labels are used ONLY inside
    calibrate_thresholds(), never in the detect functions.

SENSOR LIMITATION: the synthetic Phase 7 rollover/impact signals exceed the
+/-2 g and +/-250 deg/s ranges configured in the MPU6050 firmware. A real
sensor would saturate and clip such peaks; they are NOT clipped here so the
comparison stays reproducible. Thresholds near or above those ranges could
never fire on that hardware. This is a future hardware-realism issue.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from common import FEATURE_COLUMNS
from window_features import WINDOW_SIZE, extract_window_features

DEFAULT_CONFIG_PATH = "threshold_config_v1.json"

# Binary definition (stated, not assumed). Everything else is "no accident".
ACCIDENT_SCENARIOS = frozenset({"Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision"})
NON_ACCIDENT_SCENARIOS = frozenset({"Normal Driving", "Pothole", "Hard Braking", "Sharp Turn"})
EVENT_CLASSES = ["Normal Driving", "Pothole", "Hard Braking", "Sharp Turn",
                 "Minor Accident", "Severe Accident", "Rollover"]   # no Multi-Impact: see docstring

CALIBRATION_PERCENTILE = 99.0

ACCIDENT = "ACCIDENT"
NORMAL = "NORMAL"


@dataclass(frozen=True)
class ThresholdConfig:
    """All thresholds in one place (units in the field names)."""
    # --- accident gate (calibrated: P99 of train NON-accident windows) ---
    accel_mag_max_g: float
    gyro_mag_max_dps: float
    vibration_mean: float
    speed_drop_kmh: float
    # --- cascade noise floors (calibrated: train Normal Driving windows) ---
    turn_gyro_z_range_dps: float
    brake_accel_x_mean_g: float        # NEGATIVE: fires when accel_x_g_mean < this
    pothole_accel_z_range_g: float
    # --- author-chosen constants (see module docstring) ---
    rollover_dominance_ratio: float = 2.0
    rollover_min_gyro_x_range_dps: float = 100.0
    severe_accel_mag_g: float = 6.0
    provenance: Dict[str, Any] = field(default_factory=dict, compare=False)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ThresholdConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# (rule name, human description, feature(s) used, function features -> measured value, config field)
_GATE_RULES = [
    ("impact_acceleration", "peak acceleration magnitude in the window exceeded the threshold",
     "accel_mag_max", lambda f: f["accel_mag_max"], "accel_mag_max_g"),
    ("angular_rate", "peak gyroscope magnitude in the window exceeded the threshold",
     "gyro_mag_max", lambda f: f["gyro_mag_max"], "gyro_mag_max_dps"),
    ("vibration", "mean vibration level in the window exceeded the threshold",
     "vibration_level_mean", lambda f: f["vibration_level_mean"], "vibration_mean"),
    ("rapid_deceleration", "speed dropped across the window by more than the threshold",
     "speed_kmh_first_to_last (negated)", lambda f: -f["speed_kmh_first_to_last"], "speed_drop_kmh"),
]

REQUIRED_FEATURES = [
    "accel_mag_max", "gyro_mag_max", "vibration_level_mean", "speed_kmh_first_to_last",
    "gyro_x_dps_range", "gyro_y_dps_range", "gyro_z_dps_range", "accel_x_g_mean", "accel_z_g_range",
]


# ---------------------------------------------------------------------------
# INFERENCE (no labels, no ML)
# ---------------------------------------------------------------------------

def _validate(features: Mapping[str, float]) -> None:
    missing = [k for k in REQUIRED_FEATURES if k not in features]
    if missing:
        raise ValueError(f"features is missing required keys: {missing}")
    bad = [k for k in REQUIRED_FEATURES
           if not isinstance(features[k], (int, float, np.integer, np.floating))
           or not math.isfinite(float(features[k]))]
    if bad:
        raise ValueError(f"features contain non-numeric or non-finite values for: {bad}")


def _event_class(features: Mapping[str, float], cfg: ThresholdConfig, accident: bool) -> Dict[str, Any]:
    """Secondary decision cascade -> (label, the rules that chose it)."""
    f = {k: float(features[k]) for k in REQUIRED_FEATURES}
    if accident:
        other = max(f["gyro_y_dps_range"], f["gyro_z_dps_range"])
        if (f["gyro_x_dps_range"] >= cfg.rollover_min_gyro_x_range_dps
                and f["gyro_x_dps_range"] >= cfg.rollover_dominance_ratio * other):
            return {"label": "Rollover", "rules": ["roll_rate_dominates_other_gyro_axes"]}
        if f["accel_mag_max"] >= cfg.severe_accel_mag_g:
            return {"label": "Severe Accident", "rules": ["accel_mag_max_at_or_above_severe_level"]}
        return {"label": "Minor Accident", "rules": ["accident_below_severe_level"]}
    if f["gyro_z_dps_range"] > cfg.turn_gyro_z_range_dps:
        return {"label": "Sharp Turn", "rules": ["yaw_rate_range_above_noise_floor"]}
    if f["accel_x_g_mean"] < cfg.brake_accel_x_mean_g:
        return {"label": "Hard Braking", "rules": ["longitudinal_deceleration_above_noise_floor"]}
    if f["accel_z_g_range"] > cfg.pothole_accel_z_range_g:
        return {"label": "Pothole", "rules": ["vertical_accel_range_above_noise_floor"]}
    return {"label": "Normal Driving", "rules": []}


def detect(features: Mapping[str, float], config: Optional[ThresholdConfig] = None) -> Dict[str, Any]:
    """
    Decide from one window's feature dict.

    Returns
    -------
    {
      "event_detected": bool,            # True iff at least one gate rule fired
      "classification": "ACCIDENT"|"NORMAL",
      "triggered_rules": [rule names],
      "severity_inputs": [{"rule", "description", "feature", "value", "threshold"} ...],
      "event_class": one of EVENT_CLASSES,   # secondary, never Multi-Impact
      "event_class_rules": [...],
    }
    """
    _validate(features)
    cfg = config if config is not None else load_default_config()

    triggered: List[str] = []
    inputs: List[Dict[str, Any]] = []
    for name, description, feature_name, value_fn, cfg_field in _GATE_RULES:
        value = float(value_fn(features))
        threshold = float(getattr(cfg, cfg_field))
        if value > threshold:                      # strict: equality does not fire
            triggered.append(name)
            inputs.append({"rule": name, "description": description, "feature": feature_name,
                           "value": value, "threshold": threshold})

    accident = bool(triggered)
    cls = _event_class(features, cfg, accident)
    return {
        "event_detected": accident,
        "classification": ACCIDENT if accident else NORMAL,
        "triggered_rules": triggered,
        "severity_inputs": inputs,
        "event_class": cls["label"],
        "event_class_rules": cls["rules"],
    }


def detect_window(window_readings: Union[pd.DataFrame, Sequence[Mapping[str, float]]],
                  config: Optional[ThresholdConfig] = None) -> Dict[str, Any]:
    """detect() on a raw window of exactly WINDOW_SIZE readings (keys: common.FEATURE_COLUMNS)."""
    window_df = (window_readings if isinstance(window_readings, pd.DataFrame)
                 else pd.DataFrame(list(window_readings)))
    if len(window_df) != WINDOW_SIZE:
        raise ValueError(f"Expected exactly {WINDOW_SIZE} readings in a window, got {len(window_df)}.")
    missing = [c for c in FEATURE_COLUMNS if c not in window_df.columns]
    if missing:
        raise ValueError(f"window_readings is missing required keys: {missing}")
    return detect(extract_window_features(window_df), config)


def load_default_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> ThresholdConfig:
    """Load the calibrated config written by comparison_report.py."""
    try:
        return ThresholdConfig.from_json(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No threshold config at '{path}'. Run comparison_report.py first "
            "(it calibrates thresholds on the TRAINING recordings only)."
        )


# ---------------------------------------------------------------------------
# CALIBRATION (the only place labels are used; training windows only)
# ---------------------------------------------------------------------------

def calibrate_thresholds(train_windows: pd.DataFrame, label_column: str = "scenario") -> ThresholdConfig:
    """
    Derive threshold values from TRAINING windows only (see module docstring).
    `train_windows` is a window-feature table with a scenario column; the
    caller must pass only training-split windows (comparison_report.py does).
    """
    non_acc = train_windows[train_windows[label_column].isin(NON_ACCIDENT_SCENARIOS)]
    normal = train_windows[train_windows[label_column] == "Normal Driving"]
    if non_acc.empty or normal.empty:
        raise ValueError("train_windows must contain non-accident and Normal Driving windows")

    p = CALIBRATION_PERCENTILE
    gate = {
        "accel_mag_max_g": float(np.percentile(non_acc["accel_mag_max"], p)),
        "gyro_mag_max_dps": float(np.percentile(non_acc["gyro_mag_max"], p)),
        "vibration_mean": float(np.percentile(non_acc["vibration_level_mean"], p)),
        "speed_drop_kmh": float(np.percentile(-non_acc["speed_kmh_first_to_last"], p)),
    }
    floors = {
        "turn_gyro_z_range_dps": float(np.percentile(normal["gyro_z_dps_range"], p)),
        "brake_accel_x_mean_g": float(np.percentile(normal["accel_x_g_mean"], 100.0 - p)),
        "pothole_accel_z_range_g": float(np.percentile(normal["accel_z_g_range"], p)),
    }
    provenance = {
        "method": ("accident-gate thresholds = P%g of each rule quantity over TRAINING windows of "
                   "Normal Driving/Pothole/Hard Braking/Sharp Turn; cascade noise floors = P%g (P%g for "
                   "the negative-going braking rule) over TRAINING Normal Driving windows; "
                   "rollover_dominance_ratio, rollover_min_gyro_x_range_dps and severe_accel_mag_g are "
                   "author-chosen constants. Test set not used.") % (p, p, 100.0 - p),
        "percentile": p,
        "n_train_windows_total": int(len(train_windows)),
        "n_train_non_accident_windows": int(len(non_acc)),
        "n_train_normal_windows": int(len(normal)),
        "author_chosen": ["rollover_dominance_ratio", "rollover_min_gyro_x_range_dps", "severe_accel_mag_g"],
    }
    return ThresholdConfig(**gate, **floors, provenance=provenance)
