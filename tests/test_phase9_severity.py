"""Phase 9: severity_v2 -- levels, boundaries, multi-sensor logic, explanations, validation, independence."""

import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataset_generator_v2 as g2
import severity_v2 as sv
from common import FEATURE_COLUMNS
from severity_v2 import SeverityConfig, assess, assess_window

ROOT = Path(__file__).resolve().parent.parent

# Round-number anchors (lo=0, hi=100, so normalised value == raw value / 100 and points == weight * raw).
CFG = SeverityConfig(
    accel_lo_g=0, accel_hi_g=100, gyro_lo_dps=0, gyro_hi_dps=100, speed_loss_lo_kmh=0, speed_loss_hi_kmh=100,
    vibration_lo=0, vibration_hi=100, sustained_rotation_dps=80.0,
)
# Single-channel weighting: score == accel_mag_max, which makes band boundaries exact.
ONLY_ACCEL = SeverityConfig(
    accel_lo_g=0, accel_hi_g=100, gyro_lo_dps=0, gyro_hi_dps=100, speed_loss_lo_kmh=0, speed_loss_hi_kmh=100,
    vibration_lo=0, vibration_hi=100, sustained_rotation_dps=80.0,
    w_accel=1.0, w_gyro=0.0, w_speed=0.0, w_vibration=0.0,
)


def feats(accel=0.0, gyro=0.0, speed_loss=0.0, vib=0.0, gyro_mean=0.0, **extra):
    f = dict(accel_mag_max=accel, gyro_mag_max=gyro, speed_kmh_first_to_last=-speed_loss,
             vibration_level_mean=vib, gyro_mag_mean=gyro_mean)
    f.update(extra)
    return f


# --- the four levels ---------------------------------------------------------------------

def test_low_example():
    r = assess(feats(accel=10, gyro=10, speed_loss=10, vib=10), CFG)
    assert r["severity"] == "LOW" and r["score"] == pytest.approx(10.0)


def test_medium_example():
    r = assess(feats(accel=100, gyro=0), CFG)          # one channel at full scale = 35 points
    assert r["severity"] == "MEDIUM" and r["score"] == pytest.approx(35.0)


def test_high_example():
    r = assess(feats(accel=100, gyro=100), CFG)        # 35 + 30 = 65
    assert r["severity"] == "HIGH" and r["score"] == pytest.approx(65.0)


def test_critical_example():
    r = assess(feats(accel=100, gyro=100, speed_loss=100, vib=100), CFG)
    assert r["severity"] == "CRITICAL" and r["score"] == pytest.approx(100.0)


def test_severity_levels_constant():
    assert sv.SEVERITY_LEVELS == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# --- exact boundaries: below / at / above, lower bound inclusive --------------------------------

@pytest.mark.parametrize("edge,below,at", [(25.0, "LOW", "MEDIUM"), (50.0, "MEDIUM", "HIGH"), (75.0, "HIGH", "CRITICAL")])
def test_band_boundaries(edge, below, at):
    just_below = math.nextafter(edge, -math.inf)
    just_above = math.nextafter(edge, math.inf)
    assert assess(feats(accel=just_below), ONLY_ACCEL)["severity"] == below
    assert assess(feats(accel=edge), ONLY_ACCEL)["severity"] == at
    assert assess(feats(accel=just_above), ONLY_ACCEL)["severity"] == at


@pytest.mark.parametrize("name,kw", [("accel", "accel"), ("gyro", "gyro"), ("speed", "speed_loss"), ("vib", "vib")])
def test_channel_normalisation_clips_and_is_exact_at_anchors(name, kw):
    assert assess(feats(**{kw: -50}), CFG)["score"] == 0.0                 # below lo -> 0
    assert assess(feats(**{kw: 1e9}), CFG)["score"] == assess(feats(**{kw: 100}), CFG)["score"]   # above hi -> saturates


def test_active_channel_boundary():
    at = assess(feats(accel=25.0), CFG)["evidence"]["acceleration"]
    below = assess(feats(accel=math.nextafter(25.0, 0)), CFG)["evidence"]["acceleration"]
    assert at["active"] and not below["active"]


def test_escalation_boundaries():
    # sustained rotation must be STRICTLY above 80, score must reach 50 (inclusive)
    base = dict(accel=50.0)
    assert assess(feats(**base, gyro_mean=80.0), ONLY_ACCEL)["severity"] == "HIGH"                 # not sustained
    assert assess(feats(**base, gyro_mean=math.nextafter(80.0, 1e9)), ONLY_ACCEL)["severity"] == "CRITICAL"
    low = assess(feats(accel=math.nextafter(50.0, 0), gyro_mean=500.0), ONLY_ACCEL)                 # score < 50
    assert low["severity"] == "MEDIUM" and not low["escalated"]


# --- multi-sensor evidence ---------------------------------------------------------------------------------

def test_single_huge_spike_with_calm_other_sensors_is_not_critical_or_high():
    for cfg in (CFG, sv.load_default_config()):
        r = assess(feats(accel=1e6, gyro=0, speed_loss=0, vib=0), cfg)
        assert r["severity"] == "MEDIUM" and r["score"] <= 35.0 + 1e-9
        assert [k for k, e in r["evidence"].items() if e["active"]] == ["acceleration"]
        assert any("only one" in x for x in r["reasons"])


def test_any_single_channel_is_capped_at_medium():
    for kw in ("accel", "gyro", "speed_loss", "vib"):
        assert assess(feats(**{kw: 1e9}), CFG)["severity"] in ("LOW", "MEDIUM")


def test_more_corroborating_channels_raise_severity():
    levels = [
        assess(feats(accel=100), CFG),
        assess(feats(accel=100, gyro=100), CFG),
        assess(feats(accel=100, gyro=100, speed_loss=100), CFG),
        assess(feats(accel=100, gyro=100, speed_loss=100, vib=100), CFG),
    ]
    scores = [r["score"] for r in levels]
    assert scores == sorted(scores) and len(set(scores)) == 4
    assert [r["severity"] for r in levels] == ["MEDIUM", "HIGH", "CRITICAL", "CRITICAL"]


def test_multiple_simultaneous_indicators_are_all_reported():
    r = assess(feats(accel=90, gyro=90, speed_loss=90, vib=90), CFG)
    assert {"high_peak_acceleration", "elevated_angular_rate", "significant_speed_loss",
            "elevated_vibration"} <= set(r["triggered_rules"])
    assert sum(e["active"] for e in r["evidence"].values()) == 4


def test_rollover_like_sustained_rotation_escalates_to_critical():
    r = assess(feats(accel=60, gyro=90, speed_loss=0, vib=40, gyro_mean=120), CFG)       # score = 21+27+0+6 = 54
    assert r["score"] == pytest.approx(54.0)
    assert r["escalated"] and r["severity"] == "CRITICAL"
    assert "sustained_rotation" in r["triggered_rules"] and "rollover_escalation" in r["triggered_rules"]
    same_without = assess(feats(accel=60, gyro=90, speed_loss=0, vib=40, gyro_mean=10), CFG)
    assert same_without["severity"] == "HIGH" and not same_without["escalated"]


def test_sustained_rotation_alone_without_other_evidence_does_not_escalate():
    r = assess(feats(gyro=100, gyro_mean=500), CFG)           # score 30
    assert r["severity"] == "MEDIUM" and r["sustained_rotation"] and not r["escalated"]


@pytest.mark.parametrize("kw", ["accel", "gyro", "speed_loss", "vib", "gyro_mean"])
def test_score_is_monotonic_in_each_input(kw):
    rng = np.random.default_rng(0)
    for _ in range(50):
        base = {k: float(rng.uniform(0, 100)) for k in ("accel", "gyro", "speed_loss", "vib", "gyro_mean")}
        bumped = {**base, kw: base[kw] + float(rng.uniform(0, 50))}
        assert assess(feats(**bumped), CFG)["score"] >= assess(feats(**base), CFG)["score"] - 1e-9


# --- explanation ---------------------------------------------------------------------------------------------------

def test_output_schema_and_evidence_adds_up():
    r = assess(feats(accel=80, gyro=40, speed_loss=20, vib=10, gyro_mean=5), CFG)
    assert set(r) == {"severity", "score", "triggered_rules", "reasons", "evidence", "sustained_rotation",
                      "escalated", "scale_note"}
    assert set(r["evidence"]) == {"acceleration", "angular", "speed_loss", "vibration"}
    for e in r["evidence"].values():
        assert set(e) == {"feature", "value", "lo", "hi", "normalized", "weight", "points", "active"}
        assert 0.0 <= e["normalized"] <= 1.0
    assert sum(e["points"] for e in r["evidence"].values()) == pytest.approx(r["score"], abs=1e-2)
    assert "not" in r["scale_note"].lower() and "certified" in r["scale_note"].lower()


def test_explanation_names_the_evidence_that_raised_severity():
    r = assess(feats(accel=100, gyro=100, speed_loss=0, vib=0), CFG)
    assert r["triggered_rules"] == ["high_peak_acceleration", "elevated_angular_rate"]
    joined = " ".join(r["reasons"])
    assert "acceleration" in joined and "angular" in joined and "speed" not in joined
    assert len(r["reasons"]) >= len(r["triggered_rules"])


def test_calm_window_has_no_triggered_rules():
    r = assess(feats(accel=1, gyro=1), CFG)
    assert r["triggered_rules"] == [] and r["severity"] == "LOW"


def test_speed_gain_adds_no_severity():
    assert assess(feats(speed_loss=-500), CFG)["score"] == 0.0


# --- invalid input -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{}, None, []])
def test_empty_input_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        assess(bad, CFG)


def test_missing_key_rejected():
    f = feats()
    del f["gyro_mag_mean"]
    with pytest.raises(ValueError, match="gyro_mag_mean"):
        assess(f, CFG)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "high", None, True])
def test_bad_values_rejected(bad):
    with pytest.raises(ValueError):
        assess(feats(accel=bad), CFG)


def test_bad_window_rejected():
    w = g2.generate_recording(np.random.default_rng(0), "Pothole", 0).iloc[:10][FEATURE_COLUMNS]
    with pytest.raises(ValueError, match="exactly"):
        assess_window(w.iloc[:3], CFG)
    with pytest.raises(ValueError, match="missing"):
        assess_window(w.drop(columns=["speed_kmh"]), CFG)


@pytest.mark.parametrize("kwargs", [
    dict(accel_hi_g=0.0),                                  # hi <= lo
    dict(w_accel=0.5),                                     # weights do not sum to 1
    dict(w_accel=-0.1, w_gyro=0.75),                       # negative weight
    dict(band_medium=60.0),                                # bands not ordered
])
def test_invalid_config_rejected(kwargs):
    base = dict(accel_lo_g=0, accel_hi_g=100, gyro_lo_dps=0, gyro_hi_dps=100, speed_loss_lo_kmh=0,
                speed_loss_hi_kmh=100, vibration_lo=0, vibration_hi=100, sustained_rotation_dps=80.0)
    with pytest.raises(ValueError):
        SeverityConfig(**{**base, **kwargs})


def test_missing_default_config_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="severity_v2.py"):
        sv.load_default_config(tmp_path / "nope.json")


# --- determinism, no ground truth, no ML ---------------------------------------------------------------------------------

def test_deterministic():
    f = feats(accel=70, gyro=55, speed_loss=30, vib=20, gyro_mean=40)
    assert assess(f, CFG) == assess(dict(f), CFG)
    w = g2.generate_recording(np.random.default_rng(3), "Rollover", 0).iloc[20:30][FEATURE_COLUMNS]
    assert assess_window(w, CFG) == assess_window(w.copy(), CFG)


def test_labels_and_extra_keys_cannot_change_the_result():
    f = feats(accel=70, gyro=55)
    polluted = {**f, "scenario": "Severe Accident", "label": "Severe Accident", "predicted_label": "Rollover",
                "recording_id": 7, "confidence": 0.99}
    assert assess(polluted, CFG) == assess(f, CFG)


def test_inference_source_has_no_scenario_or_ml_reference():
    for fn in (sv.assess, sv.assess_window, sv._validate, sv.level_for_score):
        src = inspect.getsource(fn)
        assert not re.search(r"scenario|SCENARIO|label|joblib|sklearn|predict|probab|RandomForest|ACCIDENT", src), fn.__name__
        assert "label_column" not in inspect.signature(fn).parameters
    assert "label_column" in inspect.signature(sv.calibrate_severity_config).parameters


def test_no_ml_or_pipeline_imports_in_module_source():
    imports = [l for l in Path(sv.__file__).read_text(encoding="utf-8").splitlines()
               if re.match(r"\s*(import|from)\s", l) and not l.lstrip().startswith("#")]
    module_level = [l for l in imports if not l.startswith(" ")]
    assert not any(re.search(r"joblib|sklearn|predict|train_|ensemble|realtime_pipeline|threshold_detector", l)
                   for l in module_level)


def test_imports_independently_without_loading_any_ml_library():
    code = ("import sys, severity_v2; "
            "bad = [m for m in ('sklearn','joblib','predict_expanded','predict_expanded_v2','train_expanded_model',"
            "'realtime_pipeline','threshold_detector','streamlit') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --- calibration (training data only) ------------------------------------------------------------------------------------

def make_train():
    rng = np.random.default_rng(1)
    rows = []
    for sc, hi in [("Normal Driving", 1), ("Pothole", 1), ("Hard Braking", 1), ("Sharp Turn", 1),
                   ("Minor Accident", 10), ("Severe Accident", 10), ("Rollover", 10), ("Multi-Impact Collision", 10)]:
        for _ in range(100):
            rows.append(dict(scenario=sc, accel_mag_max=rng.uniform(1, 1.2) * hi, gyro_mag_max=rng.uniform(0, 100) * hi,
                             gyro_mag_mean=rng.uniform(0, 50) * hi, vibration_level_mean=rng.uniform(0, 2) * hi,
                             speed_kmh_first_to_last=-rng.uniform(0, 40)))
    return pd.DataFrame(rows)


def test_calibration_percentiles_and_sources():
    df = make_train()
    cfg = sv.calibrate_severity_config(df)
    non = df[df.scenario.isin(sv._NON_ACCIDENT_SCENARIOS)]
    acc = df[df.scenario.isin(sv._ACCIDENT_SCENARIOS)]
    assert cfg.accel_lo_g == pytest.approx(np.percentile(non.accel_mag_max, 99))
    assert cfg.accel_hi_g == pytest.approx(np.percentile(acc.accel_mag_max, 90))
    assert cfg.vibration_hi == pytest.approx(np.percentile(acc.vibration_level_mean, 90))
    assert cfg.sustained_rotation_dps == pytest.approx(np.percentile(non.gyro_mag_mean, 99))
    assert (cfg.speed_loss_lo_kmh, cfg.speed_loss_hi_kmh) == (14.0, 35.0)           # engineering assumption
    prov = cfg.provenance
    assert prov["test_set_used"] is False and "not computed" in prov["severity_accuracy"]
    assert "speed_loss_lo_kmh" in prov["engineering_assumptions"]["fields"]
    assert "accel_lo_g" in prov["training_derived"]["fields"]


def test_calibration_needs_both_groups():
    df = make_train()
    with pytest.raises(ValueError):
        sv.calibrate_severity_config(df[df.scenario == "Normal Driving"])


def test_config_json_roundtrip(tmp_path):
    cfg = sv.calibrate_severity_config(make_train())
    cfg.to_json(tmp_path / "c.json")
    assert SeverityConfig.from_json(tmp_path / "c.json") == cfg


def test_committed_config_is_reproducible_from_training_recordings_only(monkeypatch):
    import train_expanded_model_v2 as t2
    from train_expanded_model import build_windows_from_dataset
    df = pd.read_csv(ROOT / t2.DATASET_CSV_V2)
    train_df, test_df = t2.split_by_recording(df)
    saved = SeverityConfig.from_json(ROOT / sv.DEFAULT_CONFIG_PATH)
    assert saved == sv.calibrate_severity_config(build_windows_from_dataset(train_df))
    seen = {}
    real = sv.calibrate_severity_config
    monkeypatch.setattr(sv, "calibrate_severity_config",
                        lambda w, *a, **k: (seen.setdefault("ids", set(w["recording_id"])), real(w, *a, **k))[1])
    sv.calibrate_severity_config(sv._train_windows_from_phase7_split())
    assert seen["ids"].isdisjoint(set(test_df.recording_id)) and len(seen["ids"]) == 640


def test_committed_config_file_has_sources_for_every_threshold():
    raw = json.loads((ROOT / sv.DEFAULT_CONFIG_PATH).read_text())
    prov = raw["provenance"]
    declared = set(prov["training_derived"]["fields"]) | set(prov["engineering_assumptions"]["fields"])
    numeric = {k for k, v in raw.items() if k != "provenance" and isinstance(v, (int, float))}
    assert numeric == declared                          # every threshold has a documented source
    assert prov["test_set_used"] is False


# --- qualitative evaluation file (descriptive only) --------------------------------------------------------------------------

def test_qualitative_report_makes_no_accuracy_claim_and_looks_plausible():
    rep = json.loads((ROOT / "severity_qualitative_v2.json").read_text())
    assert "no severity accuracy" in rep["disclaimer"].lower() and "accuracy" not in rep["per_scenario"]
    ps = rep["per_scenario"]
    for sc in ("Normal Driving", "Pothole", "Sharp Turn"):
        assert ps[sc]["peak_window_level_counts"]["LOW"] == 30
    med = {sc: v["peak_window_score"]["median"] for sc, v in ps.items()}
    assert med["Normal Driving"] < med["Minor Accident"] < med["Severe Accident"]


# --- backward compatibility ---------------------------------------------------------------------------------------------------------

def test_existing_severity_lookup_and_state_machine_untouched():
    from realtime_pipeline import severity_for_label, next_status
    assert severity_for_label("Severe Accident") == "HIGH" and severity_for_label("Pothole") == "LOW"
    assert next_status("MONITORING", "HIGH") == "ALERT COUNTDOWN"
    assert "severity_v2" not in (ROOT / "realtime_pipeline.py").read_text(encoding="utf-8")
    # Phase 10 imports severity_v2 only to RECORD it (config path metadata + replay); the live severity
    # display and the alert state machine must still come from the legacy lookup.
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "st.session_state.severity = severity_for_label(label)" in app_src
    assert not re.search(r"severity_v2\.assess|from severity_v2 import", app_src)
