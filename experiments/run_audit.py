"""
PHASE 11.5 -- run the evaluation audit of a Phase 11 baseline.

    python -m experiments.run_audit [--baseline experiments/results/<phase11 dir>] [--out experiments/results/phase11_5_audit]

Read-only for the baseline, the model, the thresholds and every earlier phase's outputs (all hashed before and
after). Writes only into the audit output directory.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn

import predict_expanded
import threshold_detector as td
from experiments import audit as A
from window_features import WINDOW_FEATURE_COLUMNS

DEFAULT_OUT = Path(__file__).with_name("results") / "phase11_5_audit"
RESULTS_ROOT = Path(__file__).with_name("results")


def find_baseline(root: Path = RESULTS_ROOT) -> Path:
    cands = sorted(d for d in root.glob("phase11_virtual_hardware_baseline_2*") if (d / "phase11_summary.json").is_file())
    if not cands:
        raise FileNotFoundError("no Phase 11 baseline found under experiments/results")
    return cands[-1]


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def _clean(o):
    """JSON-safe copy (numpy scalars -> python, NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run_audit(baseline_dir: Path, out_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    baseline_dir, out_dir = Path(baseline_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, file=sys.stderr)) if verbose else (lambda *a: None)

    # ------------------------------------------------------------------ 1. freeze
    frozen_before, baseline_before = A.frozen_hashes(), A.tree_hash(baseline_dir)
    reported = json.loads((baseline_dir / "phase11_summary.json").read_text())
    meta, rep = reported["metadata"], reported["summary"]
    recs = pd.read_csv(baseline_dir / "recordings.csv")
    win_rep = pd.read_csv(baseline_dir / "window_results.csv")
    n_per = int(meta["runs_per_scenario"])
    log(f"baseline {baseline_dir.name}: {len(recs)} recordings, {len(win_rep)} windows")

    # ------------------------------------------------------------------ 2. independent re-simulation + window decisions
    sims = [A.resimulate(r.scenario, int(r.seed), int(r.sample_count)) for r in recs.itertuples()]
    thr_cfg = td.load_default_config()
    W = A.window_table(sims, thr_cfg)
    model = predict_expanded.load_model()
    labels, proba = A.predict_frozen(model, W)
    W["ml_label"], W["ml_bin"] = labels, np.isin(labels, ["Minor Accident", "Severe Accident"]).astype(int)
    for i, c in enumerate(model.classes_):
        W[f"ml_prob_{c}"] = proba[:, i]

    # consistency of the SAVED recordings with the re-simulation (the stream both detectors saw)
    from recording_store import RecordingStore
    store = RecordingStore(baseline_dir / "recordings")
    max_sample_diff, n_checked = 0.0, 0
    for sim in sims[:: max(1, len(sims) // 40)]:                           # a 40-recording spread of the saved files
        rec = store.load(f"p11-{sim['scenario'].lower().replace(' ', '-')}-{sim['seed']}")
        for s, t in zip(rec.samples, sim["ticks"]):
            max_sample_diff = max(max_sample_diff, max(abs(s[k] - v) for k, v in t["reading"].items()))
        n_checked += 1

    key = ["scenario", "seed", "end_tick"]
    rep_w = win_rep.rename(columns={"window_end_tick": "end_tick"})
    merged = W.merge(rep_w, on=key, suffixes=("", "_rep"), how="outer", indicator=True)
    prob_cols = [c for c in W.columns if c.startswith("ml_prob_")]
    window_consistency = {
        "windows_recomputed": int(len(W)), "windows_in_baseline": int(len(win_rep)),
        "windows_missing_either_side": int((merged["_merge"] != "both").sum()),
        "ml_label_mismatches": int((merged["ml_label"] != merged["ml_label_rep"]).sum()),
        "ml_binary_mismatches": int((merged["ml_bin"] != merged["ml_binary"]).sum()),
        "threshold_binary_mismatches": int((merged["th_binary"] != merged["threshold_binary"]).sum()),
        "phase_mismatches": int((merged["phase"] != merged["phase_rep"]).sum()),
        "max_abs_probability_difference": float(max((merged[c] - merged[c + "_rep"]).abs().max() for c in prob_cols)),
        "saved_recordings_checked": n_checked, "max_abs_sample_difference_saved_vs_resimulated": float(max_sample_diff),
    }
    log("window consistency", window_consistency)

    # ------------------------------------------------------------------ 3. metric recalculation (independent code path)
    out_ml, out_th = A.recording_outcomes(W, "ml_bin"), A.recording_outcomes(W, "th_binary")
    rows: List[Dict[str, Any]] = []

    def add(level, method, metric, recomputed, reported_v, num=None, den=None, den_def=""):
        diff = None if recomputed is None or reported_v is None else abs(float(recomputed) - float(reported_v))
        rows.append({"level": level, "method": method, "metric": metric, "recomputed": recomputed, "reported": reported_v,
                     "abs_difference": diff, "matches": None if diff is None else bool(diff < 1e-9) or (recomputed is None and reported_v is None),
                     "numerator": num, "denominator": den, "denominator_definition": den_def})

    DEN = {
        "recording": {"accuracy": "all recordings (accident + non-accident)", "precision": "recordings the detector flagged (TP+FP)",
                      "recall": "accident recordings (TP+FN)", "f1": "harmonic mean of precision and recall",
                      "false_positive_rate": "non-accident recordings (FP+TN)", "false_negative_rate": "accident recordings (FN+TP)"},
        "window": {"accuracy": "labelled windows: all windows of non-accident recordings + PRE_EVENT and EVENT windows of accident recordings (POST_EVENT excluded)",
                   "precision": "windows flagged (TP+FP)", "recall": "EVENT windows of accident recordings (TP+FN)",
                   "f1": "harmonic mean of precision and recall",
                   "false_positive_rate": "negative windows: non-accident recordings' windows + accident recordings' PRE_EVENT windows (FP+TN)",
                   "false_negative_rate": "EVENT windows of accident recordings (FN+TP)"},
    }
    computed: Dict[str, Dict[str, Any]] = {"recording_level": {}, "window_level": {}}
    for method, outc, wcol, mkey in (("ml", out_ml, "ml_bin", "ml"), ("threshold", out_th, "th_binary", "threshold")):
        c = A.conf(outc["truth"], outc["decision"])
        r = A.rates(c)
        computed["recording_level"][method] = {**c, **r}
        for k, v in c.items():
            add("recording", method, k, v, rep["recording_level"][mkey][k], den="n/a (count)")
        for k, v in r.items():
            num_den = {"accuracy": (c["TP"] + c["TN"], sum(c.values())), "precision": (c["TP"], c["TP"] + c["FP"]),
                       "recall": (c["TP"], c["TP"] + c["FN"]), "f1": (None, None),
                       "false_positive_rate": (c["FP"], c["FP"] + c["TN"]), "false_negative_rate": (c["FN"], c["FN"] + c["TP"])}[k]
            add("recording", method, k, v, rep["recording_level"][mkey][k], *num_den, DEN["recording"][k])
        lab = W[W["truth"].notna()]
        cw = A.conf(lab["truth"], lab[wcol])
        rw = A.rates(cw)
        computed["window_level"][method] = {**cw, **rw, "excluded_post_event_windows": int(W["truth"].isna().sum())}
        for k, v in cw.items():
            add("window", method, k, v, rep["window_level"][mkey][k], den="n/a (count)")
        add("window", method, "excluded_post_event_windows", int(W["truth"].isna().sum()),
            rep["window_level"][mkey]["excluded_post_event_windows"], den="n/a (count)")
        for k, v in rw.items():
            num_den = {"accuracy": (cw["TP"] + cw["TN"], sum(cw.values())), "precision": (cw["TP"], cw["TP"] + cw["FP"]),
                       "recall": (cw["TP"], cw["TP"] + cw["FN"]), "f1": (None, None),
                       "false_positive_rate": (cw["FP"], cw["FP"] + cw["TN"]), "false_negative_rate": (cw["FN"], cw["FN"] + cw["TP"])}[k]
            add("window", method, k, v, rep["window_level"][mkey][k], *num_den, DEN["window"][k])
        acc = outc[outc["truth"] == 1]
        lat = acc["latency_ms"].dropna()
        lat_rep = rep["recording_level"][mkey]["latency_ms"]
        for k, v in (("n", len(lat)), ("mean", lat.mean()), ("median", lat.median()), ("std", lat.std(ddof=1)),
                     ("min", lat.min()), ("max", lat.max())):
            add("recording", method, f"latency_ms_{k}", _num(float(v)), _num(lat_rep[k]), den="detected accident recordings",
                den_def="latency exists only for accident recordings detected at/after onset")
        add("recording", method, "pre_event_false_alarm_recordings", int(acc["pre_event_alarm"].sum()),
            rep["recording_level"][mkey]["pre_event_false_alarm_count"], int(acc["pre_event_alarm"].sum()), len(acc),
            "accident recordings (positive before true onset)")
        add("recording", method, "missed_events", int((acc["decision"] == 0).sum()), rep["recording_level"][mkey]["missed_event_count"],
            den=len(acc), den_def="accident recordings")
        computed[f"outcomes_{method}"] = outc
    pair = A.stratified_pair_counts(out_ml.sort_values(["scenario", "seed"]).reset_index(drop=True),
                                    out_th.sort_values(["scenario", "seed"]).reset_index(drop=True))
    pr = rep["paired_comparison"]["all_recordings"]
    for k in ("only_first_correct", "only_second_correct", "discordant", "n_pairs"):
        add("recording", "paired", f"mcnemar_{k}", pair[k], pr[k], den_def="same 400 recordings scored by both methods")
    add("recording", "paired", "mcnemar_p_value", pair["p_value"], pr["p_value"], den_def="exact two-sided binomial (scipy.stats.binomtest)")
    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(out_dir / "metric_recalculation.csv", index=False)
    discrepancies = metric_df[metric_df["matches"] == False]            # noqa: E712
    log(f"metric rows {len(metric_df)}, discrepancies {len(discrepancies)}")

    # ------------------------------------------------------------------ per-scenario audit
    ps_rows = []
    rep_ps = {r["scenario"]: r for r in rep["per_scenario"]}
    for sc in A.SCENARIOS:
        row: Dict[str, Any] = {"scenario": sc, "n_recordings": int((out_ml["scenario"] == sc).sum()),
                               "truth": "accident" if sc in A.ACCIDENT else "non-accident"}
        for method, outc, wcol, rkey in (("ml", out_ml, "ml_bin", "ml"), ("threshold", out_th, "th_binary", "threshold")):
            o = outc[outc["scenario"] == sc]
            rate = float(o["decision"].mean())
            kind = "detection_rate" if sc in A.ACCIDENT else "false_alarm_rate"
            row[f"{method}_{kind}_recomputed"] = rate
            row[f"{method}_{kind}_reported"] = rep_ps[sc][f"{rkey}_{kind}"]
            row[f"{method}_{kind}_ci95_wilson"] = json.dumps(A.wilson(int(o["decision"].sum()), len(o)))
            row[f"{method}_pre_event_alarm_rate_recomputed"] = float(o["pre_event_alarm"].mean()) if sc in A.ACCIDENT else None
            row[f"{method}_median_latency_ms_recomputed"] = _num(float(o["latency_ms"].median())) if o["latency_ms"].notna().any() else None
            ww = W[(W["scenario"] == sc) & W["truth"].notna()]
            row[f"{method}_window_positive_rate_recomputed"] = float(ww[wcol].mean())
            row[f"{method}_window_positive_rate_reported"] = rep_ps[sc][f"{rkey}_window_positive_rate"]
        row["ml_out_of_distribution"] = sc in ("Rollover", "Multi-Impact Collision")
        ps_rows.append(row)
    ps_df = pd.DataFrame(ps_rows)
    ps_df.to_csv(out_dir / "per_scenario_audit.csv", index=False)

    # ------------------------------------------------------------------ 4. timeline
    first_any = {m: o.set_index(["scenario", "seed"]) for m, o in (("ml", out_ml), ("threshold", out_th))}
    th_rule_first = (W[(W["th_binary"] == 1)].sort_values("end_tick").groupby(["scenario", "seed"]).first())["th_rules"]
    ml_label_first = (W[(W["ml_bin"] == 1)].sort_values("end_tick").groupby(["scenario", "seed"]).first())["ml_label"]
    tl_rows = []
    for sim, r in zip(sims, recs.itertuples()):
        t = A.timeline_row(sim, r.event_onset_tick)
        key_ = (sim["scenario"], sim["seed"])
        row = {"recording_id": r.recording_id, **t, "ground_truth": r.ground_truth}
        for m in ("ml", "threshold"):
            o = first_any[m].loc[key_]
            row[f"{m}_first_positive_any_tick"] = _num(o["first_any_tick"])
            row[f"{m}_first_eligible_tick"] = _num(o["first_eligible_tick"])
        row["threshold_first_alarm_rules"] = th_rule_first.get(key_)
        row["ml_first_alarm_class"] = ml_label_first.get(key_)
        tl_rows.append(row)
    tl = pd.DataFrame(tl_rows)
    for m in ("ml", "threshold"):
        tl[f"{m}_first_any_minus_onset_ticks"] = tl[f"{m}_first_positive_any_tick"] - tl["onset_recomputed"]
        pre = tl["speed_drop_start_tick"].combine(tl["vibration_event_start_tick"], lambda a, b: np.nanmin([a, b]) if not (pd.isna(a) and pd.isna(b)) else np.nan)
        tl["earliest_physical_precursor_tick"] = pre
        tl[f"{m}_first_any_at_or_after_precursor"] = tl[f"{m}_first_positive_any_tick"] >= pre
    tl["speed_drop_start_minus_onset_ticks"] = tl["speed_drop_start_tick"] - tl["onset_recomputed"]
    tl["vibration_start_minus_onset_ticks"] = tl["vibration_event_start_tick"] - tl["onset_recomputed"]
    tl["shock_peak_minus_onset_ticks"] = tl["shock_peak_tick"] - tl["onset_recomputed"]
    tl.to_csv(out_dir / "event_timeline_analysis.csv", index=False)
    acc_tl = tl[tl["ground_truth"] == 1]
    med = lambda s: _num(float(s.median())) if s.notna().any() else None                # noqa: E731
    timeline_summary = {}
    for sc in A.ACCIDENT:
        g = acc_tl[acc_tl["scenario"] == sc]
        pre_ml = g[g["ml_first_any_minus_onset_ticks"] < 0]
        pre_th = g[g["threshold_first_alarm_rules"].notna() & (g["threshold_first_any_minus_onset_ticks"] < 0)]
        timeline_summary[sc] = {
            "n": int(len(g)), "onset_recomputed_matches_baseline": bool(g["onset_matches_baseline"].all()),
            "median_speed_drop_start_minus_onset_ticks": med(g["speed_drop_start_minus_onset_ticks"]),
            "median_vibration_start_minus_onset_ticks": med(g["vibration_start_minus_onset_ticks"]),
            "median_shock_peak_minus_onset_ticks": med(g["shock_peak_minus_onset_ticks"]),
            "median_first_saturation_minus_onset_ticks": med(g["first_sensor_saturation_tick"] - g["onset_recomputed"]),
            "median_vehicle_crashed_flag_minus_onset_ticks": med(g["vehicle_crashed_flag_tick"] - g["onset_recomputed"]),
            "median_ml_first_positive_minus_onset_ticks": med(g["ml_first_any_minus_onset_ticks"]),
            "median_threshold_first_positive_minus_onset_ticks": med(g["threshold_first_any_minus_onset_ticks"]),
            "ml_pre_event_alarm_recordings": int(len(pre_ml)),
            "ml_pre_event_alarms_at_or_after_a_physical_precursor": int(pre_ml["ml_first_any_at_or_after_precursor"].sum()),
            "threshold_pre_event_alarm_recordings": int(len(pre_th)),
            "threshold_pre_event_alarms_at_or_after_a_physical_precursor": int(pre_th["threshold_first_any_at_or_after_precursor"].sum()),
        }
    pre_pos = W[(W["truth"] == 0) & (W["phase"] == "PRE_EVENT") & W["scenario"].isin(A.ACCIDENT)]
    rule_counts: Dict[str, int] = {}
    for s in pre_pos[pre_pos["th_binary"] == 1]["th_rules"]:
        for rname in s.split(";"):
            rule_counts[rname] = rule_counts.get(rname, 0) + 1
    timeline_evidence = {
        "per_scenario": timeline_summary,
        "threshold_rules_in_pre_event_positive_windows": rule_counts,
        "ml_classes_in_pre_event_positive_windows": pre_pos[pre_pos["ml_bin"] == 1]["ml_label"].value_counts().to_dict(),
        "constants": {"onset_shock_threshold": A.ONSET_SHOCK, "speed_drop_start_kmh_below_initial_median": A.SPEED_DROP_START_KMH,
                      "vibration_event_term": A.VIBRATION_EVENT_TERM},
        "physical_impact_time": ("NOT exposed by the simulation state. VehicleState has no impact timestamp; `shock` is a smooth "
                                 "Gaussian envelope. `shock_peak_tick` and `vehicle_crashed_flag_tick` (shock > 0.3 for impact/rollover) are "
                                 "reported as PROXIES only; no impact time has been invented."),
        "why_0.1": ("No rationale for 0.1 is recorded in the code or the Phase 11 documentation beyond 'fixed before results were "
                    "seen'. It is an arbitrary round number on a unitless envelope scaled so that 1 ~ the peak of a typical event."),
    }

    # ------------------------------------------------------------------ 6. saturation
    sat_rows = []
    for sim, r in zip(sims, recs.itertuples()):
        ticks = sim["ticks"]
        onset, end = A.onset_and_end([t["shock"] for t in ticks])
        def cnt(lo, hi, pred=lambda t: t["saturated"]):
            return sum(1 for t in ticks if lo <= t["tick"] < hi and pred(t))
        o_, e_ = (onset if onset is not None else len(ticks)), (end + 1 if end is not None else len(ticks))
        row = {"recording_id": r.recording_id, "scenario": sim["scenario"], "seed": sim["seed"], "ground_truth": int(r.ground_truth),
               "saturated_ticks_total": cnt(0, len(ticks)), "saturated_ticks_pre_event": cnt(0, o_),
               "saturated_ticks_event": cnt(o_, e_), "saturated_ticks_post_event": cnt(e_, len(ticks)),
               "max_abs_accel_physical_g": max(max(abs(t["physical"][a]) for a in ("ax", "ay", "az")) for t in ticks),
               "max_abs_gyro_physical_dps": max(max(abs(t["physical"][a]) for a in ("gx", "gy", "gz")) for t in ticks)}
        for ax in A.AXES:
            row[f"saturated_ticks_{ax}"] = cnt(0, len(ticks), lambda t, ax=ax: t["sat_axes"][ax])
        row["ml_decision"] = int(out_ml[(out_ml["scenario"] == sim["scenario"]) & (out_ml["seed"] == sim["seed"])]["decision"].iloc[0])
        row["threshold_decision"] = int(out_th[(out_th["scenario"] == sim["scenario"]) & (out_th["seed"] == sim["seed"])]["decision"].iloc[0])
        sat_rows.append(row)
    sat = pd.DataFrame(sat_rows)
    sat.to_csv(out_dir / "saturation_analysis.csv", index=False)
    acc_sat = sat[sat["ground_truth"] == 1]
    saturation_summary = {
        "recordings_with_any_saturation_by_scenario": {sc: int((sat[sat["scenario"] == sc]["saturated_ticks_total"] > 0).sum()) for sc in A.SCENARIOS},
        "accident_recordings_saturating_during_event": int((acc_sat["saturated_ticks_event"] > 0).sum()),
        "accident_recordings_total": int(len(acc_sat)),
        "accident_recordings_with_pre_event_saturation": int((acc_sat["saturated_ticks_pre_event"] > 0).sum()),
        "non_accident_recordings_with_saturation": int((sat[sat["ground_truth"] == 0]["saturated_ticks_total"] > 0).sum()),
        "axis_saturated_ticks_by_scenario": {sc: {ax: int(sat[sat["scenario"] == sc][f"saturated_ticks_{ax}"].sum()) for ax in A.AXES} for sc in A.SCENARIOS},
        "window_level_strata": A.saturation_strata(W, W["ml_bin"].to_numpy(), W["th_binary"].to_numpy()),
    }
    saturation_summary["can_isolate_saturation_effect_at_recording_level"] = bool(
        0 < saturation_summary["accident_recordings_saturating_during_event"] < saturation_summary["accident_recordings_total"])
    saturation_summary["note"] = (
        "Every accident recording clips during its event and no non-accident recording clips, so at recording level saturation is "
        "perfectly confounded with 'accident scenario'; this experiment cannot isolate its effect there. The window-level strata "
        "compare windows of the SAME accident recordings with and without a clipped sample; they are confounded with event "
        "intensity and timing, so they describe association only.") if not saturation_summary["can_isolate_saturation_effect_at_recording_level"] else (
        "Some, but not all, accident recordings clip during the event.")

    # ------------------------------------------------------------------ 4/5. shortcut diagnostics and ablations
    log("diagnostics ...")
    train, test = A.phase4_split_windows()
    train_means = train[WINDOW_FEATURE_COLUMNS].mean()
    lab_sup = W["scenario"].isin(["Hard Braking", "Minor Accident", "Normal Driving", "Pothole", "Severe Accident", "Sharp Turn"]) & W["phase"].isin(["EVENT", "NO_EVENT"])
    frozen_eval = A.eval_predictions(W, labels)
    knock = {"frozen_no_perturbation": frozen_eval}
    groups = A.FEATURE_GROUPS
    for g, cols in groups.items():
        knock[f"frozen_with_{g}_features_set_to_training_mean"] = A.eval_predictions(W, A.knockout(model, W, train_means, cols))
    for g, cols in groups.items():
        others = [c for c in WINDOW_FEATURE_COLUMNS if c not in cols]
        knock[f"frozen_keeping_only_{g}_features"] = A.eval_predictions(W, A.knockout(model, W, train_means, others))
    perm = {g: A.group_permutation_drop(model, W, cols, W["scenario"], lab_sup) for g, cols in groups.items()}
    importances = dict(zip(WINDOW_FEATURE_COLUMNS, model.feature_importances_))
    grouped_imp = {g: float(sum(importances[c] for c in cols)) for g, cols in groups.items()}
    top_features = sorted(importances.items(), key=lambda kv: -kv[1])[:10]
    diag = A.diagnostic_models(train, test, W)
    # sanity: the diagnostic 'A_full_features' model uses the frozen model's recipe and training recordings
    from sklearn.ensemble import RandomForestClassifier
    rebuilt = RandomForestClassifier(n_estimators=200, random_state=42).fit(train[WINDOW_FEATURE_COLUMNS], train["scenario"])
    recipe_identical = bool((rebuilt.predict(W[WINDOW_FEATURE_COLUMNS]) == labels).all())
    ablation = {
        "label": "DIAGNOSTIC ANALYSES. No production model, threshold or configuration was changed.",
        "frozen_model_perturbation": {
            "label": "Frozen model with some inputs replaced by their Phase 4 training mean at inference. NO retraining. Shows what the frozen model leans on, not why.",
            "results": knock},
        "frozen_model_importance": {
            "impurity_importance_grouped": grouped_imp, "impurity_importance_top10": top_features,
            "joint_group_permutation_multiclass_accuracy_drop": perm,
            "caveat": "Importance and permutation measure how much the fitted model uses an input; they do not establish causation or that the signal is physically meaningful."},
        "diagnostic_retrained_models": {
            "label": "DIAGNOSTIC RETRAINING on a recording-level split of the Phase 4 dataset (same split/hyperparameters as the frozen model: RF(200), random_state=42, "
                     "test_size=0.2 stratified by recording). Trained on Phase 4 TRAIN recordings only; evaluated on held-out Phase 4 recordings and on the Phase 11 "
                     "virtual-hardware recordings, which are separate data. These are NOT the frozen baseline and must not be compared with it as if they were.",
            "split": {"train_recordings": int(train["recording_id"].nunique()), "test_recordings": int(test["recording_id"].nunique()),
                      "train_windows": int(len(train)), "test_windows": int(len(test)), "overlap_recordings": 0},
            "diagnostic_full_model_reproduces_frozen_predictions_on_phase11_windows": recipe_identical,
            "models": diag},
        "distribution_shift_phase4_vs_phase11": A.distribution_shift(train, W),
    }
    (out_dir / "feature_ablation_report.json").write_text(json.dumps(_clean(ablation), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 7. OOD and type classification
    ev = W[W["phase"] == "EVENT"]
    ood = {}
    for sc in A.ACCIDENT:
        g = ev[ev["scenario"] == sc]
        row = {"event_windows": int(len(g)), "predicted_class_counts": g["ml_label"].value_counts().to_dict(),
               "binary_accident_detection_event_window_rate": float(g["ml_bin"].mean()),
               "recording_level_detection_rate": float(out_ml[out_ml["scenario"] == sc]["decision"].mean())}
        if sc in ("Minor Accident", "Severe Accident"):
            row["exact_type_correct_event_window_rate"] = float((g["ml_label"] == sc).mean())
            row["confused_with_other_accident_type_rate"] = float((g["ml_bin"].eq(1) & (g["ml_label"] != sc)).mean())
        else:
            row["exact_type_correct_event_window_rate"] = None
            row["type_note"] = "no ML class exists for this scenario: type accuracy is undefined, not zero"
        ood[sc] = row

    # ------------------------------------------------------------------ 8. statistics
    ci = {}
    for method in ("ml", "threshold"):
        c = computed["recording_level"][method]
        ci[method] = {"accuracy_ci95_wilson": A.wilson(c["TP"] + c["TN"], 400), "recall_ci95_wilson": A.wilson(c["TP"], c["TP"] + c["FN"]),
                      "false_positive_rate_ci95_wilson": A.wilson(c["FP"], c["FP"] + c["TN"]),
                      "precision_ci95_wilson": A.wilson(c["TP"], c["TP"] + c["FP"]),
                      "rule_of_three_fp_rate_upper_95": (3 / (c["FP"] + c["TN"]) if c["FP"] == 0 else None)}
    ok = {m: (computed[f"outcomes_{m}"].sort_values(["scenario", "seed"])).reset_index(drop=True) for m in ("ml", "threshold")}
    acc_only = lambda df: df[df["truth"] == 1].reset_index(drop=True)                          # noqa: E731
    non_only = lambda df: df[df["truth"] == 0].reset_index(drop=True)                          # noqa: E731
    stat = {
        "confidence_intervals_recording_level": ci,
        "paired_mcnemar_exact_scipy": {"all_recordings": pair,
                                       "accident_recordings": A.stratified_pair_counts(acc_only(ok["ml"]), acc_only(ok["threshold"])),
                                       "non_accident_recordings": A.stratified_pair_counts(non_only(ok["ml"]), non_only(ok["threshold"]))},
        "unit_of_inference": "recording (n=400; 200 accident, 200 non-accident). The 16,400 windows are strongly correlated within a recording and are never used as independent samples for p-values or intervals.",
        "caveat": "Every number is conditional on the synthetic generator and its seeds. Recordings of one scenario are draws from one parametric family, not a real-world population; nothing here is evidence of real-world performance.",
    }

    # ------------------------------------------------------------------ integrity
    frozen_after, baseline_after = A.frozen_hashes(), A.tree_hash(baseline_dir)
    integrity = {"frozen_files_unchanged": frozen_before == frozen_after,
                 "baseline_directory_unchanged": baseline_before["tree_sha256"] == baseline_after["tree_sha256"]}
    if not all(integrity.values()):
        raise RuntimeError(f"INTEGRITY FAILURE: {integrity}")
    manifest = {
        "audit": "phase11_5_evaluation_audit", "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_experiment_id": meta["experiment_id"], "baseline_dir": baseline_dir.name,
        "frozen_hashes_before": frozen_before, "frozen_hashes_after": frozen_after,
        "baseline_tree_sha256_before": baseline_before["tree_sha256"], "baseline_tree_sha256_after": baseline_after["tree_sha256"],
        "baseline_n_files": baseline_before["n_files"],
        "baseline_file_hashes_top_level": {k: v for k, v in baseline_before["files"].items() if "/" not in k},
        "integrity": integrity, "baseline_metadata_hashes_match_current_files": {
            "ml_model": meta["ml_model_sha256"] == frozen_before["ml_model"],
            "threshold_config": meta["threshold_config_sha256"] == frozen_before["threshold_config"]},
        "diagnostic_model_seeds": {"random_forest_random_state": 42, "split_random_state": 42, "permutation_seed": 0},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "scipy": __import__("scipy").__version__},
        "command": "python -m experiments.run_audit", "output_files": sorted(p.name for p in out_dir.iterdir()),
    }
    report = {
        "title": "Phase 11.5 evaluation audit", "baseline": meta["experiment_id"],
        "reproduction": {"metric_rows_compared": int(len(metric_df)), "discrepancies": int(len(discrepancies)),
                         "discrepant_rows": discrepancies.head(20).to_dict("records"),
                         "all_metrics_reproduced": bool(len(discrepancies) == 0), "window_consistency": window_consistency,
                         "computed_recording_level": computed["recording_level"], "computed_window_level": computed["window_level"]},
        "timeline": timeline_evidence, "saturation": saturation_summary, "ood": ood, "statistics": stat,
        "shortcut_summary": {
            "frozen_perturbation": {k: {"recording_level_accuracy": v["recording_level"]["accuracy"], "recording_false_positive_rate": v["recording_level"]["false_positive_rate"],
                                        "recording_recall": v["recording_level"]["recall"], "multiclass_accuracy": v["multiclass_accuracy_supported_event_windows"]}
                                    for k, v in knock.items()},
            "diagnostic_models": {k: {"phase4_heldout_multiclass_accuracy": v["phase4_heldout_test"]["multiclass_accuracy"],
                                      "phase11_recording_accuracy": v["phase11_virtual_hardware"]["recording_level"]["accuracy"],
                                      "phase11_recording_fpr": v["phase11_virtual_hardware"]["recording_level"]["false_positive_rate"],
                                      "phase11_recording_recall": v["phase11_virtual_hardware"]["recording_level"]["recall"]}
                                  for k, v in diag.items()},
            "diagnostic_full_model_reproduces_frozen": recipe_identical},
        "integrity": integrity,
    }
    (out_dir / "audit_report.json").write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    (out_dir / "reproducibility_manifest.json").write_text(json.dumps(_clean(manifest), indent=2), encoding="utf-8")
    from experiments import audit_markdown
    (out_dir / "audit_report.md").write_text(audit_markdown.build(_clean(report), _clean(ablation), _clean(manifest)), encoding="utf-8")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="PHASE 11.5 evaluation audit (read-only for the baseline).")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    baseline = Path(a.baseline) if a.baseline else find_baseline()
    rep = run_audit(baseline, Path(a.out))
    print(json.dumps({"discrepancies": rep["reproduction"]["discrepancies"], "integrity": rep["integrity"]}, indent=2))


if __name__ == "__main__":
    main()
