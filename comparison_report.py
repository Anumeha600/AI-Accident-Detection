"""
PHASE 8 -- ML (Phase 7) vs traditional threshold detector, on the SAME data.
----------------------------------------------------------------------------
Research question: does multi-sensor machine-learning classification
discriminate accidents/events better than a conventional threshold detector?

PROTOCOL (fixed in advance; nothing is tuned against the test set)
  1. Load expanded_synthetic_dataset_v2.csv and reproduce the Phase 7
     recording-level split with train_expanded_model_v2.split_by_recording()
     -- the identical 640 train / 160 test recordings and the identical
     3200 / 800 windows the ML model was trained/evaluated on.
  2. Calibrate the threshold detector on TRAINING windows only
     (threshold_detector.calibrate_thresholds) and save the result to
     threshold_config_v1.json. The test windows are not touched until step 3.
  3. Score the held-out test windows once: the ML model (Phase 7 .joblib,
     loaded read-only) and the threshold detector both receive the same
     36 window features and nothing else (no labels).
  4. Report binary ACCIDENT-vs-NON-ACCIDENT results (primary) at window level
     and at recording level (a recording is flagged if any of its windows is),
     plus the 8-class secondary comparison, detection time, and limitations.

BINARY DEFINITION (explicit, see threshold_detector.ACCIDENT_SCENARIOS)
  accident     = Minor Accident, Severe Accident, Rollover, Multi-Impact Collision
  non-accident = Normal Driving, Pothole, Hard Braking, Sharp Turn
  Pothole / hard braking / sharp turn are real driving events but NOT
  accidents; flagging them is a false positive. For the ML model, its
  8-class prediction is mapped to this binary with the same definition.
  Every window carries its recording's label (Phase 4 methodology), so windows
  from before an event starts are "positives" that look like normal driving;
  this caps window-level recall for BOTH detectors and is why the
  recording-level view (did we flag the accident recording at all?) is also
  reported.

Outputs: comparison_report.json, comparison_report.csv, threshold_config_v1.json.
No existing dataset, model, or evaluation file is read for writing.

IMPORTANT: ALL DATA IS SIMULATED/SYNTHETIC. These numbers compare two
detectors on one hand-written simulator; they are not estimates of real-world
performance, and the simulator's author also wrote the threshold recipe.
"""

import json
import math
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import threshold_detector as td
import train_expanded_model_v2 as t2
from common import LABEL_COLUMN
from dataset_generator import SAMPLE_RATE_HZ
from window_features import STRIDE, WINDOW_FEATURE_COLUMNS, WINDOW_SIZE

REPORT_JSON = "comparison_report.json"
REPORT_CSV = "comparison_report.csv"
THRESHOLD_CONFIG_PATH = td.DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, Any]:
    """Accident = positive. Returns counts and rates (NaN where undefined)."""
    t, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, tn = int((t & p).sum()), int((~t & ~p).sum())
    fp, fn = int((~t & p).sum()), int((t & ~p).sum())
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if tp else (0.0 if (fp or fn) else float("nan"))
    return {
        "n": int(len(t)), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": _safe_div(tp + tn, len(t)), "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": _safe_div(fp, fp + tn), "false_negative_rate": _safe_div(fn, fn + tp),
        "confusion_matrix": {"labels": ["non-accident", "accident"], "rows_actual_cols_predicted": [[tn, fp], [fn, tp]]},
    }


def mcnemar_exact_p(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, Any]:
    """Exact two-sided McNemar test on paired correctness (windows from one recording are not independent)."""
    b = int((correct_a & ~correct_b).sum())   # A right, B wrong
    c = int((~correct_a & correct_b).sum())   # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> Dict[str, Any]:
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "per_class": {l: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for l, p, r, f, s in zip(labels, prec, rec, f1, sup)},
        "confusion_matrix": {"labels": labels, "rows_actual_cols_predicted": cm.tolist()},
    }


def detection_latency(test_windows: pd.DataFrame, flagged: np.ndarray) -> Dict[str, Any]:
    """
    Time from the START of an accident recording to the END of its first flagged
    window (seconds). NOT referenced to the true event onset: onset is not stored
    in the dataset, so this is a time-to-first-alert bound with 1 s granularity,
    including the 1 s needed to fill a window.
    """
    df = test_windows.assign(flagged=flagged)
    acc = df[df[LABEL_COLUMN].isin(td.ACCIDENT_SCENARIOS)]
    times, missed = [], 0
    for _, g in acc.groupby("recording_id"):
        hit = g[g["flagged"]]
        if hit.empty:
            missed += 1
        else:
            times.append((int(hit["window_start_row"].min()) + WINDOW_SIZE) / SAMPLE_RATE_HZ)
    out: Dict[str, Any] = {
        "reference": "seconds from recording start to end of first flagged window (not event-onset referenced)",
        "n_accident_recordings": int(acc["recording_id"].nunique()),
        "n_detected": len(times), "n_never_detected": missed,
    }
    out.update({"mean_s": float(np.mean(times)), "median_s": float(np.median(times)),
                "min_s": float(np.min(times)), "max_s": float(np.max(times))} if times else {})
    return out


def _per_scenario_flag_rate(test_windows: pd.DataFrame, flagged: np.ndarray) -> Dict[str, Dict[str, float]]:
    df = test_windows.assign(flagged=flagged)
    out = {}
    for sc, g in df.groupby(LABEL_COLUMN):
        out[sc] = {"is_accident_class": sc in td.ACCIDENT_SCENARIOS, "n_windows": int(len(g)),
                   "windows_flagged_accident": int(g["flagged"].sum()),
                   "window_flag_rate": float(g["flagged"].mean()),
                   "recordings_flagged_any_window": int(g.groupby("recording_id")["flagged"].any().sum()),
                   "n_recordings": int(g["recording_id"].nunique()),
                   "recording_flag_rate": float(g.groupby("recording_id")["flagged"].any().mean())}
    return out


def pre_event_diagnostic(test_windows: pd.DataFrame, ml_binary: np.ndarray, th_binary: np.ndarray) -> Dict[str, Any]:
    """
    Descriptive check (NOT used for tuning): how often is each detector's
    ACCIDENT flag raised on the FIRST window (samples 0-9) of each test
    recording? In the generators the event is centred at sample ~12 or later
    (Multi-Impact may start earlier, so treat it with care), so for the other
    accident scenarios this window is mostly event-free. A high flag rate there
    means the detector is reacting to something other than the event itself,
    e.g. the higher baseline sensor noise / speed range of the accident
    generators, which is a SIMULATOR ARTIFACT.
    """
    first = (test_windows["window_start_row"] == 0).to_numpy()
    df = test_windows[first].assign(ml=ml_binary[first], threshold=th_binary[first])
    out = {sc: {"is_accident_class": sc in td.ACCIDENT_SCENARIOS, "n_windows": int(len(g)),
                "ml_flag_rate": float(g["ml"].mean()), "threshold_flag_rate": float(g["threshold"].mean())}
           for sc, g in df.groupby(LABEL_COLUMN)}
    return {"window": "first window of each test recording (window_start_row == 0, samples 0-9)",
            "interpretation": ("Accident-class rates near 1.0 for ML with few events in the window indicate the model "
                               "recognises generator fingerprints (baseline noise/speed), not accident events. "
                               "Multi-Impact's first impact can fall inside this window."),
            "per_scenario": out}


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------

def run_comparison() -> Dict[str, Any]:
    df = pd.read_csv(t2.DATASET_CSV_V2)
    train_df, test_df = t2.split_by_recording(df)                 # asserts disjoint recording IDs
    train_w, test_w = t2.window_split(train_df, test_df)          # asserts no window on both sides
    train_ids, test_ids = set(train_w["recording_id"]), set(test_w["recording_id"])
    assert train_ids.isdisjoint(test_ids)

    # --- 2. calibrate on TRAIN ONLY, before the test windows are scored ---
    config = td.calibrate_thresholds(train_w)
    config.to_json(THRESHOLD_CONFIG_PATH)

    # --- 3. score the held-out windows, one feature row each, no labels ---
    X_test = test_w[WINDOW_FEATURE_COLUMNS]
    y_true = test_w[LABEL_COLUMN].to_numpy()

    model = joblib.load(t2.EXPANDED_V2_MODEL_PATH)               # read-only
    ml_pred = model.predict(X_test)

    th_results = [td.detect(row, config) for row in X_test.to_dict("records")]
    th_binary = np.array([r["event_detected"] for r in th_results])
    th_class = np.array([r["event_class"] for r in th_results])

    # consistency: this is the same test set the Phase 7 evaluation used
    with open(t2.EVALUATION_REPORT_PATH, encoding="utf-8") as fh:
        phase7 = json.load(fh)
    ml_acc = float((ml_pred == y_true).mean())
    assert math.isclose(ml_acc, phase7["accuracy"], abs_tol=1e-12), "ML accuracy differs from Phase 7 report"
    assert len(test_w) == phase7["windowing"]["test_windows"]

    is_acc_true = np.isin(y_true, list(td.ACCIDENT_SCENARIOS))
    ml_binary = np.isin(ml_pred, list(td.ACCIDENT_SCENARIOS))

    # --- window-level binary (primary) ---
    win = {"machine_learning": binary_metrics(is_acc_true, ml_binary),
           "threshold": binary_metrics(is_acc_true, th_binary)}
    win["mcnemar_exact_ml_vs_threshold"] = mcnemar_exact_p(ml_binary == is_acc_true, th_binary == is_acc_true)
    win["mcnemar_note"] = ("a = ML, b = threshold. Windows within one recording are correlated, so the "
                           "p-value is optimistic; treat as a rough guide.")

    # --- recording-level binary ---
    rec_ids = test_w["recording_id"].to_numpy()
    rec_table = pd.DataFrame({"recording_id": rec_ids, "true": is_acc_true,
                              "ml": ml_binary, "th": th_binary}).groupby("recording_id").agg(
        true=("true", "first"), ml=("ml", "any"), th=("th", "any"))
    rec = {"machine_learning": binary_metrics(rec_table["true"], rec_table["ml"]),
           "threshold": binary_metrics(rec_table["true"], rec_table["th"]),
           "mcnemar_exact_ml_vs_threshold": mcnemar_exact_p((rec_table["ml"] == rec_table["true"]).to_numpy(),
                                                            (rec_table["th"] == rec_table["true"]).to_numpy())}

    labels8 = sorted(phase7["per_class"])
    ml8 = multiclass_metrics(y_true, ml_pred, labels8)
    th8 = multiclass_metrics(y_true, th_class, labels8)

    report: Dict[str, Any] = {
        "disclaimer": ("ALL DATA IS SIMULATED/SYNTHETIC. Comparison of two detectors on one hand-written "
                       "simulator; not a measure of real-world performance."),
        "research_question": ("Does multi-sensor ML classification provide better accident/event "
                              "discrimination than a conventional threshold-based detector?"),
        "dataset": {
            "csv": t2.DATASET_CSV_V2, "rows": int(len(df)), "n_recordings": int(df["recording_id"].nunique()),
            "samples_per_recording": int(df.groupby("recording_id").size().iloc[0]),
            "sample_rate_hz": SAMPLE_RATE_HZ, "window_size": WINDOW_SIZE, "stride": STRIDE,
            "split": {"level": "recording_id, before windowing (identical to Phase 7)",
                      "train_recordings": len(train_ids), "test_recordings": len(test_ids),
                      "recording_id_overlap": len(train_ids & test_ids),
                      "train_windows": int(len(train_w)), "test_windows": int(len(test_w))},
            "test_windows_per_class": {k: int(v) for k, v in pd.Series(y_true).value_counts().sort_index().items()},
        },
        "binary_definition": {
            "accident": sorted(td.ACCIDENT_SCENARIOS), "non_accident": sorted(td.NON_ACCIDENT_SCENARIOS),
            "rationale": ("An accident is a collision/rollover. Pothole, hard braking and sharp turn are "
                          "ordinary driving events; flagging them is a false positive. Window labels are "
                          "inherited from the recording, so pre-event windows of accident recordings count "
                          "as positives for both detectors."),
        },
        "thresholds": {
            "values": {k: v for k, v in vars(config).items() if k != "provenance"} if hasattr(config, "__dict__")
            else {},
            "rules": [{"rule": n, "description": d, "feature": f, "config_field": c, "comparison": "strictly greater than"}
                      for n, d, f, _, c in td._GATE_RULES],
            "provenance": config.provenance,
            "config_file": THRESHOLD_CONFIG_PATH,
            "multi_impact_note": "The threshold cascade can never output Multi-Impact Collision (single-window rules).",
        },
        "ml_model": {"path": t2.EXPANDED_V2_MODEL_PATH, "type": "RandomForestClassifier", **t2.MODEL_PARAMS,
                     "accuracy_matches_phase7_report": True},
        "binary_window_level": win,
        "binary_recording_level": rec,
        "per_scenario_accident_flag_rates": {
            "machine_learning": _per_scenario_flag_rate(test_w, ml_binary),
            "threshold": _per_scenario_flag_rate(test_w, th_binary)},
        "pre_event_window_diagnostic": pre_event_diagnostic(test_w, ml_binary, th_binary),
        "detection_latency": {"machine_learning": detection_latency(test_w, ml_binary),
                              "threshold": detection_latency(test_w, th_binary)},
        "eight_class": {"machine_learning": ml8, "threshold": th8,
                        "note": "Secondary. The threshold cascade cannot emit Multi-Impact Collision."},
        "limitations": [
            "All data is synthetic from one hand-written generator; train/test share that generator.",
            "The author wrote both the simulator and the threshold recipe; thresholds are not blind to it.",
            "Window labels are inherited from the recording, so pre-event windows are labelled positive.",
            "Detection latency is time from recording start to the end of the first flagged 1 s window; "
            "event onset is not stored, so true onset-referenced latency is not measured.",
            "Synthetic rollover/impact signals exceed the MPU6050 firmware ranges (+/-2 g, +/-250 deg/s); "
            "a real sensor would saturate and clip. Not clipped here (reproducibility); future hardware-realism work.",
            "In STM32 hardware mode vibration and speed are placeholders, which would disable two of the four rules.",
            "SIMULATOR CONFOUND: accident generators use higher baseline sensor noise and different speed ranges "
            "than non-accident ones, so ML can flag accident recordings from windows that contain no event "
            "(see pre_event_window_diagnostic). Its perfect binary score therefore overstates event discrimination.",
            "Windows from one recording are correlated; McNemar p-values are rough guides.",
            "160 test recordings (20 per class): rates carry sizeable sampling uncertainty.",
        ],
    }
    return report


def _csv_rows(report: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for level in ("binary_window_level", "binary_recording_level"):
        for det in ("machine_learning", "threshold"):
            m = report[level][det]
            rows.append({"section": level, "detector": det, "item": "accident_vs_non_accident",
                         **{k: m[k] for k in ("n", "TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "f1",
                                              "false_positive_rate", "false_negative_rate")}})
    for det in ("machine_learning", "threshold"):
        for sc, m in report["eight_class"][det]["per_class"].items():
            rows.append({"section": "eight_class_per_class", "detector": det, "item": sc,
                         "n": m["support"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
        rows.append({"section": "eight_class_overall", "detector": det, "item": "accuracy/macro_f1",
                     "accuracy": report["eight_class"][det]["accuracy"], "f1": report["eight_class"][det]["macro_f1"]})
    return pd.DataFrame(rows)


def _print_summary(r: Dict[str, Any]) -> None:
    d = r["dataset"]["split"]
    print("=" * 74)
    print("PHASE 8: ML (Phase 7) vs THRESHOLD detector -- SYNTHETIC data only")
    print("=" * 74)
    print(f"Test set: {d['test_recordings']} recordings / {d['test_windows']} windows "
          f"(train {d['train_recordings']} / {d['train_windows']}); recording-ID overlap: {d['recording_id_overlap']}")
    print("\nThresholds (calibrated on TRAIN only):")
    for k, v in r["thresholds"]["values"].items():
        print(f"  {k:32s} {v}")
    for level, title in (("binary_window_level", "WINDOW level"), ("binary_recording_level", "RECORDING level")):
        print(f"\nBINARY accident vs non-accident -- {title}")
        print(f"{'':18s}{'TP':>5}{'TN':>6}{'FP':>5}{'FN':>5}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'FPR':>8}{'FNR':>8}")
        for det in ("machine_learning", "threshold"):
            m = r[level][det]
            print(f"{det:18s}{m['TP']:>5}{m['TN']:>6}{m['FP']:>5}{m['FN']:>5}{m['accuracy']:>8.3f}{m['precision']:>8.3f}"
                  f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['false_positive_rate']:>8.3f}{m['false_negative_rate']:>8.3f}")
        mc = r[level]["mcnemar_exact_ml_vs_threshold"]
        print(f"  McNemar exact: ML-right/TH-wrong={mc['a_right_b_wrong']}, ML-wrong/TH-right={mc['a_wrong_b_right']}, "
              f"p={mc['p_value']:.4g}")
    print("\nDetection time (s from recording start to end of first flagged window; not onset-referenced):")
    for det, m in r["detection_latency"].items():
        print(f"  {det:18s} detected {m['n_detected']}/{m['n_accident_recordings']}  "
              f"mean={m.get('mean_s', float('nan')):.2f} median={m.get('median_s', float('nan')):.2f}")
    print("\nPer-scenario fraction of RECORDINGS flagged ACCIDENT (any window):")
    for sc in sorted(r["per_scenario_accident_flag_rates"]["threshold"]):
        a = r["per_scenario_accident_flag_rates"]["machine_learning"][sc]
        b = r["per_scenario_accident_flag_rates"]["threshold"][sc]
        tag = "ACC" if a["is_accident_class"] else "non"
        print(f"  [{tag}] {sc:24s} ML {a['recording_flag_rate']:.2f}   threshold {b['recording_flag_rate']:.2f}")
    print("\nDIAGNOSTIC -- accident flag rate on the FIRST window (samples 0-9, mostly before any event):")
    for sc, m in r["pre_event_window_diagnostic"]["per_scenario"].items():
        tag = "ACC" if m["is_accident_class"] else "non"
        print(f"  [{tag}] {sc:24s} ML {m['ml_flag_rate']:.2f}   threshold {m['threshold_flag_rate']:.2f}")
    print("\n8-class (secondary):")
    for det in ("machine_learning", "threshold"):
        m = r["eight_class"][det]
        print(f"  {det:18s} accuracy {m['accuracy']:.3f}   macro-F1 {m['macro_f1']:.3f}")
    print("\nAll figures are SYNTHETIC-data results; see 'limitations' in comparison_report.json.")


def main() -> None:
    report = run_comparison()
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _csv_rows(report).to_csv(REPORT_CSV, index=False)
    _print_summary(report)
    print(f"\nSaved: {REPORT_JSON}, {REPORT_CSV}, {THRESHOLD_CONFIG_PATH}")


if __name__ == "__main__":
    main()
