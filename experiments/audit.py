"""
PHASE 11.5 -- evaluation audit of the Phase 11 baseline (read-only with respect to the baseline).

Everything here is DIAGNOSTIC. Nothing is tuned, the production model and thresholds are only read, and any
model that is trained is a clearly separate DIAGNOSTIC model trained on a recording-level split of the
Phase 4 synthetic dataset (the same split the frozen model used) and never on Phase 11 data.

The metric recomputation deliberately does NOT call experiments/evaluation.py: it starts from the raw
per-window rows plus a fresh re-simulation of every recording (vehicle state, onset, phases) and uses
numpy / pandas / scipy for the arithmetic, so a bug in the Phase 11 evaluation code cannot hide itself.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import predict_expanded
import threshold_detector as td
from common import FEATURE_COLUMNS
from virtual_hardware import VirtualHardwareSensorSource
from virtual_hardware.vibration_sensor import VibrationSensor
from window_features import WINDOW_FEATURE_COLUMNS, WINDOW_SIZE, extract_window_features

ONSET_SHOCK = 0.1                  # the ORIGINAL Phase 11 definition, re-implemented here (not imported) and not changed
ACCIDENT = ("Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision")
NON_ACCIDENT = ("Normal Driving", "Pothole", "Hard Braking", "Sharp Turn")
SCENARIOS = NON_ACCIDENT + ACCIDENT
SPEED_DROP_START_KMH = 2.0         # descriptive timeline constants (they label events; nothing is tuned on them)
VIBRATION_EVENT_TERM = 0.1
AXES = ("ax", "ay", "az", "gx", "gy", "gz")
FEATURE_GROUPS = {
    "speed": [c for c in WINDOW_FEATURE_COLUMNS if c.startswith("speed_kmh_")],
    "accel": [c for c in WINDOW_FEATURE_COLUMNS if c.startswith("accel_")],
    "gyro": [c for c in WINDOW_FEATURE_COLUMNS if c.startswith("gyro_")],
    "vibration": [c for c in WINDOW_FEATURE_COLUMNS if c.startswith("vibration_level_")],
}


# ---------------------------------------------------------------------------- hashing
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(directory) -> Dict[str, Any]:
    """sha256 of every file under `directory` plus one digest over (relative path, file hash) pairs."""
    directory = Path(directory)
    files = {str(p.relative_to(directory)).replace("\\", "/"): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    digest = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in files.items()).encode()).hexdigest()
    return {"n_files": len(files), "tree_sha256": digest, "files": files}


def frozen_hashes() -> Dict[str, str]:
    paths = {"ml_model": predict_expanded.EXPANDED_MODEL_PATH, "threshold_config": td.DEFAULT_CONFIG_PATH,
             "severity_config": "severity_config_v2.json", "phase8_comparison_report_json": "comparison_report.json",
             "phase8_comparison_report_csv": "comparison_report.csv", "phase7_v2_model": "accident_classifier_model_expanded_v2.joblib",
             "phase7_dataset": "expanded_synthetic_dataset.csv", "phase7_v2_dataset": "expanded_synthetic_dataset_v2.csv",
             "phase7_v2_evaluation": "evaluation_report_expanded_v2.json"}
    return {k: sha256(p) for k, p in paths.items() if Path(p).is_file()}


# ---------------------------------------------------------------------------- re-simulation
def resimulate(scenario: str, seed: int, n: int = 50) -> Dict[str, Any]:
    """Re-run one recording through the virtual-hardware path and keep every per-tick internal state."""
    src = VirtualHardwareSensorSource(seed=seed, run_ticks=n)
    src.start(scenario)
    vib = VibrationSensor(None)                                           # only .gain_for() is used (no randomness)
    ticks = []
    for t in range(n):
        reading = src.read()
        veh = src.last_tick_result.vehicle
        mpu = src.rig.mpu
        ticks.append({"tick": t, "shock": float(veh.shock), "speed": float(veh.speed_kmh), "kind": veh.event_kind,
                      "state": veh.vehicle_state, "crashed": bool(veh.crashed),
                      "vib_event": float(vib.gain_for(veh) * veh.shock), "saturated": bool(mpu.saturated),
                      "sat_axes": dict(mpu.saturated_axes), "physical": dict(mpu.last_physical), "reading": reading})
    return {"scenario": scenario, "seed": seed, "ticks": ticks}


def first_tick(ticks: Sequence[Dict[str, Any]], pred) -> Optional[int]:
    return next((t["tick"] for t in ticks if pred(t)), None)


def onset_and_end(shocks: Sequence[float]) -> Tuple[Optional[int], Optional[int]]:
    idx = [i for i, v in enumerate(shocks) if v > ONSET_SHOCK]
    return (idx[0], idx[-1]) if idx else (None, None)


def phase_of(end: int, onset: Optional[int], event_end: Optional[int]) -> str:
    if onset is None:
        return "NO_EVENT"
    if end < onset:
        return "PRE_EVENT"
    return "POST_EVENT" if end - WINDOW_SIZE + 1 > event_end else "EVENT"


def window_truth(scenario: str, phase: str) -> Optional[int]:
    if scenario in NON_ACCIDENT:
        return 0
    return {"EVENT": 1, "PRE_EVENT": 0}.get(phase)


def timeline_row(sim: Dict[str, Any], baseline_onset: Optional[int]) -> Dict[str, Any]:
    """Timing of every physical/sensor milestone of one recording (None where the state does not expose it)."""
    ticks = sim["ticks"]
    shocks = [t["shock"] for t in ticks]
    onset, end = onset_and_end(shocks)
    base_speed = float(np.median([t["speed"] for t in ticks[:5]]))
    peak = int(np.argmax(shocks)) if max(shocks) > 0 else None
    return {
        "scenario": sim["scenario"], "seed": sim["seed"], "onset_recomputed": onset, "event_end_recomputed": end,
        "onset_matches_baseline": onset == (None if baseline_onset is None or pd.isna(baseline_onset) else int(baseline_onset)),
        "speed_drop_start_tick": first_tick(ticks, lambda t: t["speed"] < base_speed - SPEED_DROP_START_KMH),
        "vibration_event_start_tick": first_tick(ticks, lambda t: t["vib_event"] >= VIBRATION_EVENT_TERM),
        "shock_first_nonzero_tick": first_tick(ticks, lambda t: t["shock"] > 1e-3),
        "shock_peak_tick": peak, "shock_peak_value": max(shocks),
        "vehicle_crashed_flag_tick": first_tick(ticks, lambda t: t["crashed"]),
        "first_sensor_saturation_tick": first_tick(ticks, lambda t: t["saturated"]),
        "physical_impact_tick_available": False,
    }


# ---------------------------------------------------------------------------- independent window decisions
def window_table(sims: Sequence[Dict[str, Any]], threshold_cfg: td.ThresholdConfig) -> pd.DataFrame:
    """One row per sliding window: features, phase, saturation flag and the threshold decision, all recomputed."""
    rows = []
    for sim in sims:
        ticks = sim["ticks"]
        onset, end = onset_and_end([t["shock"] for t in ticks])
        df = pd.DataFrame([t["reading"] for t in ticks])
        for e in range(WINDOW_SIZE - 1, len(ticks)):
            win = df.iloc[e - WINDOW_SIZE + 1:e + 1]
            feats = extract_window_features(win)
            ph = phase_of(e, onset, end)
            th = td.detect(feats, threshold_cfg)
            rows.append({"scenario": sim["scenario"], "seed": sim["seed"], "end_tick": e, "phase": ph,
                         "truth": window_truth(sim["scenario"], ph),
                         "window_saturated": any(t["saturated"] for t in ticks[e - WINDOW_SIZE + 1:e + 1]),
                         "th_binary": int(th["event_detected"]), "th_rules": ";".join(th["triggered_rules"]),
                         "onset": onset, "event_end": end, **feats})
    return pd.DataFrame(rows)


def predict_frozen(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen model, read-only, batch prediction. Returns (labels, probability matrix)."""
    proba = model.predict_proba(X[WINDOW_FEATURE_COLUMNS])
    return model.classes_[proba.argmax(axis=1)], proba


# ---------------------------------------------------------------------------- independent metrics
def conf(truth: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    truth, pred = np.asarray(truth, dtype=int), np.asarray(pred, dtype=int)
    return {"TP": int(((truth == 1) & (pred == 1)).sum()), "TN": int(((truth == 0) & (pred == 0)).sum()),
            "FP": int(((truth == 0) & (pred == 1)).sum()), "FN": int(((truth == 1) & (pred == 0)).sum())}


def rates(c: Dict[str, int]) -> Dict[str, Optional[float]]:
    d = lambda a, b: None if b == 0 else a / b                                    # noqa: E731
    p, r = d(c["TP"], c["TP"] + c["FP"]), d(c["TP"], c["TP"] + c["FN"])
    return {"accuracy": d(c["TP"] + c["TN"], sum(c.values())), "precision": p, "recall": r,
            "f1": None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r),
            "false_positive_rate": d(c["FP"], c["FP"] + c["TN"]), "false_negative_rate": d(c["FN"], c["FN"] + c["TP"])}


def recording_outcomes(win: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Per-recording outcome of one decision column, from raw window rows + re-simulated onsets."""
    out = []
    for (sc, seed), g in win.groupby(["scenario", "seed"], sort=False):
        pos = g[g[flag] == 1]["end_tick"].to_numpy()
        onset = g["onset"].iloc[0]
        acc = sc in ACCIDENT
        if acc:
            elig = pos[pos >= onset]
            first = int(elig.min()) if len(elig) else None
            out.append({"scenario": sc, "seed": seed, "truth": 1, "decision": int(first is not None),
                        "latency_ms": None if first is None else (first - int(onset)) * 100,
                        "pre_event_alarm": bool((pos < onset).any()), "first_eligible_tick": first,
                        "first_any_tick": int(pos.min()) if len(pos) else None})
        else:
            out.append({"scenario": sc, "seed": seed, "truth": 0, "decision": int(len(pos) > 0), "latency_ms": None,
                        "pre_event_alarm": False, "first_eligible_tick": None, "first_any_tick": int(pos.min()) if len(pos) else None})
    return pd.DataFrame(out)


def wilson(k: int, n: int, z: float = 1.959964) -> Optional[List[float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> Dict[str, Any]:
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    return {"only_first_correct": b, "only_second_correct": c, "discordant": b + c, "p_value": p, "n_pairs": int(len(a_ok))}


def stratified_pair_counts(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, Any]:
    ok_a, ok_b = (a["decision"] == a["truth"]).to_numpy(), (b["decision"] == b["truth"]).to_numpy()
    return mcnemar(ok_a, ok_b)


# ---------------------------------------------------------------------------- diagnostics: shortcut / ablation
def phase4_split_windows():
    """Same recording-level split as the frozen model's training script (test_size=0.2, random_state=42, stratified)."""
    from sklearn.model_selection import train_test_split
    from train_expanded_model import build_windows_from_dataset
    df = pd.read_csv("expanded_synthetic_dataset.csv")
    recs = df[["recording_id", "scenario"]].drop_duplicates()
    train_ids, test_ids = train_test_split(recs["recording_id"], test_size=0.2, random_state=42, stratify=recs["scenario"])
    train_ids, test_ids = set(train_ids), set(test_ids)
    assert train_ids.isdisjoint(test_ids)
    train = build_windows_from_dataset(df[df["recording_id"].isin(train_ids)])
    test = build_windows_from_dataset(df[df["recording_id"].isin(test_ids)])
    assert not set(train["recording_id"]) & set(test["recording_id"])
    return train, test


def eval_predictions(win: pd.DataFrame, labels: np.ndarray, accident_classes: Sequence[str] = ("Minor Accident", "Severe Accident"),
                     supported: Sequence[str] = ("Hard Braking", "Minor Accident", "Normal Driving", "Pothole", "Severe Accident", "Sharp Turn")) -> Dict[str, Any]:
    """Window-level and recording-level scoring of ANY per-window class predictions on the Phase 11 windows."""
    w = win.copy()
    w["pred_label"] = np.asarray(labels)
    w["pred_bin"] = w["pred_label"].isin(accident_classes).astype(int)
    lab = w[w["truth"].notna()]
    wl = conf(lab["truth"].to_numpy(), lab["pred_bin"].to_numpy())
    rec = recording_outcomes(w, "pred_bin")
    per = {}
    for sc in SCENARIOS:
        r = rec[rec["scenario"] == sc]
        per[sc] = float(r["decision"].mean())                         # detection rate (accident) or false-alarm rate (non-accident)
    mc = w[(w["scenario"].isin(supported)) & (w["phase"].isin(["EVENT", "NO_EVENT"]))]
    return {"window_level": {**wl, **rates(wl)},
            "recording_level": {**conf(rec["truth"].to_numpy(), rec["decision"].to_numpy()),
                                **rates(conf(rec["truth"].to_numpy(), rec["decision"].to_numpy()))},
            "per_scenario_detection_or_false_alarm_rate": per,
            "multiclass_accuracy_supported_event_windows": float((mc["pred_label"] == mc["scenario"]).mean()) if len(mc) else None,
            "n_multiclass_windows": int(len(mc))}


def knockout(model, win: pd.DataFrame, train_means: pd.Series, replace_cols: Sequence[str]) -> np.ndarray:
    """Frozen-model diagnostic: set `replace_cols` to their Phase 4 TRAINING mean at inference. No retraining."""
    X = win[WINDOW_FEATURE_COLUMNS].copy()
    for c in replace_cols:
        X[c] = train_means[c]
    return model.predict(X)


def group_permutation_drop(model, win: pd.DataFrame, group_cols: Sequence[str], labels: pd.Series, mask: pd.Series,
                           repeats: int = 5, seed: int = 0) -> Dict[str, float]:
    """Drop in multiclass accuracy when a feature group's rows are shuffled jointly (association, not causation)."""
    X = win.loc[mask, WINDOW_FEATURE_COLUMNS]
    y = labels[mask].to_numpy()
    base = float((model.predict(X) == y).mean())
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(repeats):
        Xp = X.copy()
        perm = rng.permutation(len(Xp))
        Xp[list(group_cols)] = Xp[list(group_cols)].to_numpy()[perm]
        drops.append(base - float((model.predict(Xp) == y).mean()))
    return {"baseline_accuracy": base, "mean_accuracy_drop": float(np.mean(drops)), "std_accuracy_drop": float(np.std(drops))}


def diagnostic_models(train: pd.DataFrame, test: pd.DataFrame, win: pd.DataFrame) -> Dict[str, Any]:
    """DIAGNOSTIC RETRAINING (never the production model): RF(200, rs=42) on feature subsets, recording-level split."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    subsets = {
        "A_full_features": WINDOW_FEATURE_COLUMNS,
        "B_without_speed": [c for c in WINDOW_FEATURE_COLUMNS if c not in FEATURE_GROUPS["speed"]],
        "C_accel_only": FEATURE_GROUPS["accel"], "D_gyro_only": FEATURE_GROUPS["gyro"],
        "E_speed_only": FEATURE_GROUPS["speed"], "F_vibration_only": FEATURE_GROUPS["vibration"],
        "G_accel_plus_gyro": FEATURE_GROUPS["accel"] + FEATURE_GROUPS["gyro"],
    }
    out = {}
    for name, cols in subsets.items():
        m = RandomForestClassifier(n_estimators=200, random_state=42).fit(train[cols], train["scenario"])
        pt = m.predict(test[cols])
        bt_true = test["scenario"].isin(["Minor Accident", "Severe Accident"]).astype(int)
        bt_pred = pd.Series(pt).isin(["Minor Accident", "Severe Accident"]).astype(int)
        p11 = m.predict(win[cols])
        out[name] = {
            "n_features": len(cols), "label": "DIAGNOSTIC retrained model, NOT the frozen baseline",
            "phase4_heldout_test": {"windows": int(len(test)), "recordings": int(test["recording_id"].nunique()),
                                    "multiclass_accuracy": float((pt == test["scenario"]).mean()),
                                    "multiclass_macro_f1": float(f1_score(test["scenario"], pt, average="macro")),
                                    "binary_accident_accuracy": float((bt_true.to_numpy() == bt_pred.to_numpy()).mean())},
            "phase11_virtual_hardware": eval_predictions(win, p11),
        }
    return out


# ---------------------------------------------------------------------------- distribution shift
def distribution_shift(train: pd.DataFrame, win: pd.DataFrame, top: int = 12) -> Dict[str, Any]:
    """Phase 4 TRAINING windows vs Phase 11 windows of the same scenario (scenarios the model knows), per feature."""
    rows = []
    for sc in ("Normal Driving", "Pothole", "Hard Braking", "Sharp Turn", "Minor Accident", "Severe Accident"):
        a = train[train["scenario"] == sc]
        b = win[(win["scenario"] == sc) & (win["phase"].isin(["EVENT", "NO_EVENT"]))]
        for c in WINDOW_FEATURE_COLUMNS:
            sd = a[c].std()
            rows.append({"scenario": sc, "feature": c, "phase4_mean": float(a[c].mean()), "phase11_mean": float(b[c].mean()),
                         "standardised_shift": float((b[c].mean() - a[c].mean()) / sd) if sd > 0 else None})
    df = pd.DataFrame(rows)
    big = df.reindex(df["standardised_shift"].abs().sort_values(ascending=False).index).head(top)
    speed = train.groupby("scenario")["speed_kmh_mean"].agg(["min", "max", "mean"]).round(2)
    return {"largest_standardised_shifts": big.round(3).to_dict("records"),
            "phase4_speed_mean_by_scenario": {k: v for k, v in speed.to_dict("index").items()}}


def saturation_strata(win: pd.DataFrame, ml_bin: np.ndarray, th_bin: np.ndarray) -> Dict[str, Any]:
    """Window-level detection on EVENT windows of accident recordings, split by whether the window contains a clipped sample."""
    w = win.copy()
    w["ml"], w["th"] = ml_bin, th_bin
    ev = w[(w["truth"] == 1)]
    out = {}
    for flag, g in ev.groupby("window_saturated"):
        out["saturated_window" if flag else "unsaturated_window"] = {
            "n_windows": int(len(g)), "ml_recall": float(g["ml"].mean()), "threshold_recall": float(g["th"].mean())}
    neg = w[(w["truth"] == 0)]
    for flag, g in neg.groupby("window_saturated"):
        out["negative_" + ("saturated_window" if flag else "unsaturated_window")] = {
            "n_windows": int(len(g)), "ml_false_positive_rate": float(g["ml"].mean()),
            "threshold_false_positive_rate": float(g["th"].mean())}
    return out
