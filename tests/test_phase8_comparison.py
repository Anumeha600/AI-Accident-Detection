"""Phase 8: comparison_report -- metric helpers, fairness (same split, train-only calibration), report schema."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import comparison_report as cr
import threshold_detector as td
import train_expanded_model_v2 as t2

ROOT = Path(__file__).resolve().parent.parent
EIGHT = sorted(td.ACCIDENT_SCENARIOS | td.NON_ACCIDENT_SCENARIOS)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def report():
    return json.loads((ROOT / cr.REPORT_JSON).read_text(encoding="utf-8"))


# --- metric helpers with hand-computed values ---------------------------------------

def test_binary_metrics_known_values():
    t = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    m = cr.binary_metrics(t, p)
    assert (m["TP"], m["TN"], m["FP"], m["FN"]) == (3, 5, 1, 1)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["false_positive_rate"] == pytest.approx(1 / 6)
    assert m["false_negative_rate"] == pytest.approx(0.25)
    assert m["confusion_matrix"]["rows_actual_cols_predicted"] == [[5, 1], [1, 3]]


def test_binary_metrics_degenerate_cases_do_not_crash():
    m = cr.binary_metrics([0, 0], [0, 0])
    assert m["TN"] == 2 and math.isnan(m["precision"]) and math.isnan(m["recall"])
    m = cr.binary_metrics([1, 1], [0, 0])
    assert m["FN"] == 2 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_mcnemar_exact():
    a = np.array([True] * 10 + [False] * 0)
    b = np.array([False] * 10)
    r = cr.mcnemar_exact_p(a, b)
    assert (r["a_right_b_wrong"], r["a_wrong_b_right"]) == (10, 0)
    assert r["p_value"] == pytest.approx(2 / 2 ** 10)
    assert cr.mcnemar_exact_p(a, a)["p_value"] == 1.0


def test_detection_latency_known_values():
    w = pd.DataFrame({
        "recording_id": [0] * 5 + [1] * 5 + [2] * 5,
        "window_start_row": [0, 10, 20, 30, 40] * 3,
        "scenario": ["Rollover"] * 10 + ["Pothole"] * 5,
    })
    flagged = np.array([0, 0, 1, 1, 0] + [0] * 5 + [1] * 5, bool)
    lat = cr.detection_latency(w, flagged)
    assert lat["n_accident_recordings"] == 2 and lat["n_detected"] == 1 and lat["n_never_detected"] == 1
    assert lat["mean_s"] == pytest.approx((20 + 10) / 10)     # window ends at row 30 -> 3.0 s at 10 Hz
    assert "not event-onset referenced" in lat["reference"]


# --- committed report: schema and internal consistency -------------------------------------

def test_report_has_all_required_sections(report):
    for key in ["disclaimer", "research_question", "dataset", "binary_definition", "thresholds", "ml_model",
                "binary_window_level", "binary_recording_level", "per_scenario_accident_flag_rates",
                "pre_event_window_diagnostic", "detection_latency", "eight_class", "limitations"]:
        assert key in report, key
    assert "SYNTHETIC" in report["disclaimer"]
    assert any("MPU6050" in l and "saturate" in l for l in report["limitations"])
    assert any("SIMULATOR CONFOUND" in l for l in report["limitations"])


def test_report_dataset_and_binary_definition(report):
    s = report["dataset"]["split"]
    assert (s["train_recordings"], s["test_recordings"], s["recording_id_overlap"]) == (640, 160, 0)
    assert (s["train_windows"], s["test_windows"]) == (3200, 800)
    assert sorted(report["binary_definition"]["accident"]) == sorted(td.ACCIDENT_SCENARIOS)
    assert sorted(report["binary_definition"]["non_accident"]) == sorted(td.NON_ACCIDENT_SCENARIOS)


@pytest.mark.parametrize("level,n", [("binary_window_level", 800), ("binary_recording_level", 160)])
@pytest.mark.parametrize("det", ["machine_learning", "threshold"])
def test_binary_counts_are_consistent(report, level, n, det):
    m = report[level][det]
    assert m["TP"] + m["TN"] + m["FP"] + m["FN"] == m["n"] == n
    assert m["accuracy"] == pytest.approx((m["TP"] + m["TN"]) / n)
    assert m["false_positive_rate"] == pytest.approx(m["FP"] / (m["FP"] + m["TN"]))
    assert m["false_negative_rate"] == pytest.approx(m["FN"] / (m["FN"] + m["TP"]))
    assert m["TP"] + m["FN"] == (400 if n == 800 else 80)             # same positives for both detectors


def test_both_detectors_scored_on_the_same_positives_and_negatives(report):
    for level in ("binary_window_level", "binary_recording_level"):
        a, b = report[level]["machine_learning"], report[level]["threshold"]
        assert (a["TP"] + a["FN"], a["TN"] + a["FP"]) == (b["TP"] + b["FN"], b["TN"] + b["FP"])


def test_eight_class_matrices_cover_every_test_window(report):
    for det in ("machine_learning", "threshold"):
        cm = np.array(report["eight_class"][det]["confusion_matrix"]["rows_actual_cols_predicted"])
        assert cm.shape == (8, 8) and cm.sum() == 800
        assert report["eight_class"][det]["confusion_matrix"]["labels"] == EIGHT
    th = np.array(report["eight_class"]["threshold"]["confusion_matrix"]["rows_actual_cols_predicted"])
    assert th[:, EIGHT.index("Multi-Impact Collision")].sum() == 0        # threshold can never predict it


def test_ml_numbers_match_the_phase7_report(report):
    p7 = json.loads((ROOT / t2.EVALUATION_REPORT_PATH).read_text())
    assert report["eight_class"]["machine_learning"]["accuracy"] == pytest.approx(p7["accuracy"])


def test_latency_is_labelled_honestly(report):
    for det in ("machine_learning", "threshold"):
        lat = report["detection_latency"][det]
        assert "not event-onset referenced" in lat["reference"]
        assert lat["n_detected"] + lat["n_never_detected"] == lat["n_accident_recordings"] == 80


def test_csv_matches_json(report):
    csv = pd.read_csv(ROOT / cr.REPORT_CSV)
    row = csv[(csv.section == "binary_window_level") & (csv.detector == "threshold")].iloc[0]
    assert int(row.TP) == report["binary_window_level"]["threshold"]["TP"]
    assert {"section", "detector", "item"} <= set(csv.columns)


# --- fairness: thresholds are train-only and reproducible ---------------------------------------

def test_saved_thresholds_equal_calibration_on_training_windows_only():
    df = pd.read_csv(ROOT / t2.DATASET_CSV_V2)
    train_df, test_df = t2.split_by_recording(df)
    train_w, test_w = t2.window_split(train_df, test_df)
    saved = td.ThresholdConfig.from_json(ROOT / td.DEFAULT_CONFIG_PATH)
    assert saved == td.calibrate_thresholds(train_w)
    assert set(train_w.recording_id).isdisjoint(set(test_w.recording_id))
    assert saved != td.calibrate_thresholds(pd.concat([train_w, test_w]))      # test data would change them


def test_run_comparison_only_receives_train_windows_for_calibration(monkeypatch, tmp_path):
    seen = {}
    real = td.calibrate_thresholds

    def spy(train_windows, *a, **k):
        seen["ids"] = set(train_windows["recording_id"])
        return real(train_windows, *a, **k)

    monkeypatch.setattr(td, "calibrate_thresholds", spy)
    monkeypatch.setattr(cr, "THRESHOLD_CONFIG_PATH", str(tmp_path / "cfg.json"))
    rep = cr.run_comparison()
    df = pd.read_csv(ROOT / t2.DATASET_CSV_V2)
    _, test_df = t2.split_by_recording(df)
    assert seen["ids"].isdisjoint(set(test_df.recording_id))
    assert len(seen["ids"]) == 640
    assert rep["binary_window_level"]["threshold"]["TP"] >= 0


@pytest.mark.slow
def test_run_comparison_is_reproducible_and_leaves_protected_files_alone(monkeypatch, tmp_path, report):
    protected = [*ROOT.glob("*.joblib"), *ROOT.glob("*.csv"), ROOT / t2.EVALUATION_REPORT_PATH,
                 ROOT / "common.py", ROOT / "window_features.py"]
    protected = [p for p in protected if p.name not in (cr.REPORT_CSV,)]
    before = {p: md5(p) for p in protected}
    monkeypatch.setattr(cr, "THRESHOLD_CONFIG_PATH", str(tmp_path / "cfg.json"))
    fresh = json.loads(json.dumps(cr.run_comparison()))                 # normalise NaN/tuples via JSON
    fresh["thresholds"]["config_file"] = report["thresholds"]["config_file"]
    assert fresh == report
    assert {p: md5(p) for p in protected} == before
