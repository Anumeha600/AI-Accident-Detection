"""Phase 8: threshold_detector -- every rule, boundaries, explanations, validation, no ground truth."""

import inspect
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataset_generator_v2 as g2
import threshold_detector as td
from common import FEATURE_COLUMNS
from threshold_detector import ThresholdConfig, detect, detect_window

# Simple round-number config so boundary tests are exact.
CFG = ThresholdConfig(
    accel_mag_max_g=2.0, gyro_mag_max_dps=100.0, vibration_mean=3.0, speed_drop_kmh=20.0,
    turn_gyro_z_range_dps=5.0, brake_accel_x_mean_g=-0.1, pothole_accel_z_range_g=0.2,
)


def features(**over):
    """A benign 'normal driving' feature dict (all required keys), with overrides."""
    f = dict(accel_mag_max=1.05, gyro_mag_max=3.0, vibration_level_mean=0.5, speed_kmh_first_to_last=0.0,
             gyro_x_dps_range=3.0, gyro_y_dps_range=3.0, gyro_z_dps_range=3.0,
             accel_x_g_mean=0.0, accel_z_g_range=0.05)
    f.update(over)
    return f


# --- the four accident-gate rules: below / exactly at / just above ----------------

GATE = [
    ("impact_acceleration", "accel_mag_max", 2.0, lambda v: v),
    ("angular_rate", "gyro_mag_max", 100.0, lambda v: v),
    ("vibration", "vibration_level_mean", 3.0, lambda v: v),
    ("rapid_deceleration", "speed_kmh_first_to_last", 20.0, lambda v: -v),   # feature is negated
]


@pytest.mark.parametrize("rule,feat,thr,to_feature", GATE)
def test_rule_below_at_and_above_threshold(rule, feat, thr, to_feature):
    below = detect(features(**{feat: to_feature(math.nextafter(thr, -math.inf))}), CFG)
    at = detect(features(**{feat: to_feature(thr)}), CFG)
    above = detect(features(**{feat: to_feature(math.nextafter(thr, math.inf))}), CFG)
    assert rule not in below["triggered_rules"] and not below["event_detected"]
    assert rule not in at["triggered_rules"] and not at["event_detected"]      # strict '>'
    assert above["triggered_rules"] == [rule] and above["event_detected"]
    assert above["classification"] == "ACCIDENT" and below["classification"] == "NORMAL"


@pytest.mark.parametrize("rule,feat,thr,to_feature", GATE)
def test_each_rule_explains_itself(rule, feat, thr, to_feature):
    r = detect(features(**{feat: to_feature(thr * 2)}), CFG)
    (item,) = r["severity_inputs"]
    assert item["rule"] == rule
    assert item["value"] == pytest.approx(thr * 2) and item["threshold"] == thr
    assert item["description"] and item["feature"]


def test_speed_gain_never_triggers_deceleration_rule():
    assert not detect(features(speed_kmh_first_to_last=+500.0), CFG)["event_detected"]


# --- windows / combinations -----------------------------------------------------------

def test_normal_window_is_not_an_event():
    r = detect(features(), CFG)
    assert r == {"event_detected": False, "classification": "NORMAL", "triggered_rules": [],
                 "severity_inputs": [], "event_class": "Normal Driving", "event_class_rules": []}


def test_accident_window_triggers():
    r = detect(features(accel_mag_max=9.0, gyro_mag_max=190.0, vibration_level_mean=8.0), CFG)
    assert r["event_detected"] and r["classification"] == "ACCIDENT"


def test_multiple_simultaneous_triggers_are_all_reported_in_rule_order():
    r = detect(features(accel_mag_max=9.0, gyro_mag_max=190.0, vibration_level_mean=8.0,
                        speed_kmh_first_to_last=-40.0), CFG)
    assert r["triggered_rules"] == ["impact_acceleration", "angular_rate", "vibration", "rapid_deceleration"]
    assert [i["rule"] for i in r["severity_inputs"]] == r["triggered_rules"]
    assert all(i["value"] > i["threshold"] for i in r["severity_inputs"])


def test_one_rule_is_enough_binary_is_an_or():
    for rule, feat, thr, to_feature in GATE:
        assert detect(features(**{feat: to_feature(thr + 1)}), CFG)["event_detected"]


# --- secondary event_class cascade ---------------------------------------------------------

@pytest.mark.parametrize("over,label", [
    (dict(gyro_z_dps_range=5.01), "Sharp Turn"),
    (dict(accel_x_g_mean=-0.11), "Hard Braking"),
    (dict(accel_z_g_range=0.21), "Pothole"),
    (dict(gyro_z_dps_range=5.0, accel_x_g_mean=-0.1, accel_z_g_range=0.2), "Normal Driving"),   # at floors
    (dict(accel_mag_max=3.0), "Minor Accident"),
    (dict(accel_mag_max=6.0), "Severe Accident"),                                              # >= 6 g
    (dict(accel_mag_max=5.99), "Minor Accident"),
    (dict(accel_mag_max=3.0, gyro_mag_max=400.0, gyro_x_dps_range=400.0, gyro_y_dps_range=100.0,
          gyro_z_dps_range=100.0), "Rollover"),
    (dict(accel_mag_max=8.0, gyro_mag_max=300.0, gyro_x_dps_range=150.0, gyro_y_dps_range=120.0,
          gyro_z_dps_range=150.0), "Severe Accident"),                                         # no axis dominance
])
def test_event_class_cascade(over, label):
    assert detect(features(**over), CFG)["event_class"] == label


def test_cascade_cannot_output_multi_impact():
    assert "Multi-Impact Collision" not in td.EVENT_CLASSES
    rng = np.random.default_rng(0)
    for i in range(30):
        w = g2.generate_recording(rng, "Multi-Impact Collision", i).iloc[:10][FEATURE_COLUMNS]
        assert detect_window(w, CFG)["event_class"] != "Multi-Impact Collision"


def test_pothole_hard_braking_sharp_turn_are_not_accidents_by_definition():
    assert td.NON_ACCIDENT_SCENARIOS == {"Normal Driving", "Pothole", "Hard Braking", "Sharp Turn"}
    assert td.ACCIDENT_SCENARIOS == {"Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision"}
    assert not (td.ACCIDENT_SCENARIOS & td.NON_ACCIDENT_SCENARIOS)


# --- raw windows ------------------------------------------------------------------------------

def window(scenario, seed=0, start=15):
    return g2.generate_recording(np.random.default_rng(seed), scenario, 0).iloc[start:start + 10][FEATURE_COLUMNS]


def test_detect_window_flags_a_synthetic_severe_impact_and_not_normal_driving():
    cfg = ThresholdConfig(accel_mag_max_g=1.5, gyro_mag_max_dps=120, vibration_mean=2.5, speed_drop_kmh=50,
                          turn_gyro_z_range_dps=5, brake_accel_x_mean_g=-0.04, pothole_accel_z_range_g=0.15)
    def impact_window(seed):      # 10 samples centred on the largest acceleration deviation
        rec = g2.generate_recording(np.random.default_rng(seed), "Severe Accident", 0)
        peak = int(np.argmax(np.abs(rec["accel_x_g"].to_numpy())))
        start = min(max(peak - 4, 0), len(rec) - 10)
        return rec.iloc[start:start + 10][FEATURE_COLUMNS]

    hits = sum(detect_window(impact_window(s), cfg)["event_detected"] for s in range(10))
    false = sum(detect_window(window("Normal Driving", s), cfg)["event_detected"] for s in range(10))
    assert hits == 10 and false == 0


def test_detect_window_accepts_list_of_dicts():
    w = window("Pothole")
    assert detect_window(w.to_dict("records"), CFG) == detect_window(w, CFG)


# --- invalid input -------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{}, None])
def test_empty_or_none_features_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        detect(bad, CFG)


def test_missing_feature_key_rejected():
    f = features()
    del f["gyro_mag_max"]
    with pytest.raises(ValueError, match="gyro_mag_max"):
        detect(f, CFG)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "high", None])
def test_non_finite_or_non_numeric_value_rejected(bad):
    with pytest.raises(ValueError):
        detect(features(accel_mag_max=bad), CFG)


def test_window_with_wrong_length_or_missing_columns_rejected():
    with pytest.raises(ValueError, match="exactly"):
        detect_window(window("Pothole").iloc[:5], CFG)
    with pytest.raises(ValueError, match="exactly"):
        detect_window(pd.DataFrame(columns=FEATURE_COLUMNS), CFG)
    with pytest.raises(ValueError, match="missing"):
        detect_window(window("Pothole").drop(columns=["speed_kmh"]), CFG)


# --- determinism / no ground truth / no ML --------------------------------------------------------

def test_deterministic():
    f = features(accel_mag_max=7.0, gyro_mag_max=150.0)
    assert detect(f, CFG) == detect(dict(f), CFG)
    w = window("Rollover", 3, 20)
    assert detect_window(w, CFG) == detect_window(w.copy(), CFG)


def test_result_ignores_label_like_extra_keys():
    f = features(accel_mag_max=7.0)
    polluted = {**f, "scenario": "Normal Driving", "label": "Normal Driving", "recording_id": 1}
    assert detect(polluted, CFG) == detect(f, CFG)


def test_inference_code_never_references_labels_or_ml():
    inference = [td.detect, td.detect_window, td._event_class, td._validate]
    for fn in inference:
        src = inspect.getsource(fn)
        assert not re.search(r"scenario|LABEL|label_column|ACCIDENT_SCENARIOS|predict|joblib|sklearn", src), fn.__name__
        assert "label_column" not in inspect.signature(fn).parameters
    mod_src = Path(td.__file__).read_text(encoding="utf-8")
    code_lines = [l for l in mod_src.splitlines() if re.match(r"\s*(import|from)\s", l)]
    assert not any(re.search(r"joblib|sklearn|predict|train_|ensemble", l) for l in code_lines)


def test_only_calibration_uses_labels():
    assert "label_column" in inspect.signature(td.calibrate_thresholds).parameters


# --- calibration ---------------------------------------------------------------------------------------

def make_train_windows():
    rows = []
    rng = np.random.default_rng(0)
    for sc in ["Normal Driving", "Pothole", "Hard Braking", "Sharp Turn", "Rollover"]:
        for _ in range(200):
            rows.append(dict(scenario=sc, accel_mag_max=rng.uniform(1, 1.2) if sc != "Rollover" else 50.0,
                             gyro_mag_max=rng.uniform(0, 100), vibration_level_mean=rng.uniform(0, 2),
                             speed_kmh_first_to_last=-rng.uniform(0, 40), gyro_z_dps_range=rng.uniform(0, 4),
                             accel_x_g_mean=rng.normal(0, 0.01), accel_z_g_range=rng.uniform(0, 0.1)))
    return pd.DataFrame(rows)


def test_calibration_uses_percentile_of_non_accident_windows_only():
    df = make_train_windows()
    cfg = td.calibrate_thresholds(df)
    non = df[df.scenario != "Rollover"]
    assert cfg.accel_mag_max_g == pytest.approx(np.percentile(non.accel_mag_max, 99))
    assert cfg.accel_mag_max_g < 1.3            # the accident class (50 g) did not influence it
    assert cfg.speed_drop_kmh == pytest.approx(np.percentile(-non.speed_kmh_first_to_last, 99))
    normal = df[df.scenario == "Normal Driving"]
    assert cfg.brake_accel_x_mean_g == pytest.approx(np.percentile(normal.accel_x_g_mean, 1))
    assert cfg.provenance["n_train_windows_total"] == len(df)


def test_calibration_requires_normal_and_non_accident_windows():
    with pytest.raises(ValueError):
        td.calibrate_thresholds(make_train_windows().query("scenario == 'Rollover'"))


def test_config_json_roundtrip(tmp_path):
    cfg = td.calibrate_thresholds(make_train_windows())
    cfg.to_json(tmp_path / "c.json")
    assert ThresholdConfig.from_json(tmp_path / "c.json") == cfg


def test_missing_default_config_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="comparison_report.py"):
        td.load_default_config(tmp_path / "nope.json")
