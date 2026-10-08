"""
Phase 11 -- controlled experiment runner. Small deterministic fixtures only: the 400-recording baseline is
run manually (python -m experiments.experiment_runner), never inside pytest.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

import predict_expanded
import replay as rp
import threshold_detector as td
from common import FEATURE_COLUMNS
from experiments import evaluation as ev
from experiments import experiment_runner as er
from experiments import schemas as S
from recording_store import RecordingStore
from virtual_hardware import VirtualHardwareRig, VirtualHardwareSensorSource
from window_features import WINDOW_SIZE

DEV_RUNS = 2


@pytest.fixture(scope="module")
def dev_cfg():
    cfg = er.load_config()
    cfg["runs_per_scenario"] = DEV_RUNS
    return cfg


@pytest.fixture(scope="module")
def dev_experiment(dev_cfg, tmp_path_factory):
    before = er.hash_frozen()
    out = er.run_experiment(dev_cfg, tmp_path_factory.mktemp("p11"), experiment_id="p11_dev_fixture")
    assert er.hash_frozen() == before
    return out


def load_summary(out: Path):
    return json.loads((out / "phase11_summary.json").read_text())


def rec_row(**kw):
    base = {"scenario": "Severe Accident", "ground_truth": 1, "saturated_in_event": 0,
            "ml_binary_decision": 1, "ml_latency_ms": 0, "ml_pre_event_false_alarm": False,
            "threshold_binary_decision": 1, "threshold_latency_ms": 0, "threshold_pre_event_false_alarm": False}
    base.update(kw)
    return base


# 1 ------------------------------------------------------------------ deterministic seeds
def test_seed_schedule_is_deterministic_disjoint_and_matches_the_plan():
    cfg = er.load_config()
    sched = er.seed_schedule(cfg)
    assert sched == er.seed_schedule(er.load_config())
    assert len(sched) == 400 and len({s for _, s in sched}) == 400
    firsts = {sc: min(s for x, s in sched if x == sc) for sc in cfg["scenarios"]}
    assert firsts["Normal Driving"] == 11000 and firsts["Pothole"] == 11100 and firsts["Multi-Impact Collision"] == 11700
    assert max(s for x, s in sched if x == "Normal Driving") == 11049


def test_same_seed_gives_identical_recording_and_windows_different_seed_does_not():
    thr = td.load_default_config()
    kw = dict(experiment_id="t", n_samples=50, ml_accident_classes=["Minor Accident", "Severe Accident"], threshold_cfg=thr)
    a = er.run_recording("Severe Accident", 11500, **kw)
    b = er.run_recording("Severe Accident", 11500, **kw)
    c = er.run_recording("Severe Accident", 11501, **kw)
    assert a == b
    assert a[1] != c[1]


# 2 ------------------------------------------------------------------ ground truth mapping
def test_scenario_ground_truth_mapping_is_explicit_and_matches_phase8():
    assert set(S.ACCIDENT_SCENARIOS) == set(td.ACCIDENT_SCENARIOS)
    assert set(S.ACCIDENT_SCENARIOS) == {"Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision"}
    assert set(S.NON_ACCIDENT_SCENARIOS) == {"Normal Driving", "Pothole", "Hard Braking", "Sharp Turn"}
    assert set(S.NON_ACCIDENT_SCENARIOS) == set(td.NON_ACCIDENT_SCENARIOS)
    assert set(S.GROUND_TRUTH) == set(S.ALL_SCENARIOS) and len(S.ALL_SCENARIOS) == 8
    assert all(S.GROUND_TRUTH[s] == 1 for s in S.ACCIDENT_SCENARIOS) and all(S.GROUND_TRUTH[s] == 0 for s in S.NON_ACCIDENT_SCENARIOS)


# 3 ------------------------------------------------------------------ event onset
def test_event_span_from_vehicle_shock():
    assert er.event_span([0, 0, 0.05, 0.1, 0.11, 0.9, 0.3, 0.2, 0.05, 0]) == (4, 7)       # strict ">"
    assert er.event_span([0.0] * 10) == (None, None)
    assert er.event_span([0.5] * 3) == (0, 2)


def test_phase_assignment():
    assert er.window_phase(9, 15, 25) == S.PRE_EVENT
    assert er.window_phase(15, 15, 25) == S.EVENT and er.window_phase(30, 15, 25) == S.EVENT     # touches event end
    assert er.window_phase(34, 15, 25) == S.EVENT and er.window_phase(35, 15, 25) == S.POST_EVENT
    assert er.window_phase(20, None, None) == S.NO_EVENT


@pytest.mark.parametrize("scenario", ["Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision"])
def test_accident_onset_is_taken_from_the_virtual_vehicle_state(scenario):
    row, windows = er.run_recording(scenario, 11400, "t", 50, ["Minor Accident", "Severe Accident"], td.load_default_config())
    onset = row["event_onset_tick"]
    assert onset is not None and row["event_onset_ms"] == onset * 100 and onset >= 0
    rig = VirtualHardwareRig(11400, scenario)                    # independent replay of the vehicle
    shocks = [rig.tick().vehicle.shock for _ in range(50)]
    assert shocks[onset] > S.ONSET_SHOCK_THRESHOLD and all(v <= S.ONSET_SHOCK_THRESHOLD for v in shocks[:onset])
    assert {w["phase"] for w in windows} <= {S.PRE_EVENT, S.EVENT, S.POST_EVENT}


def test_normal_driving_has_no_event():
    row, windows = er.run_recording("Normal Driving", 11000, "t", 50, ["Minor Accident", "Severe Accident"], td.load_default_config())
    assert row["event_onset_tick"] is None and row["ground_truth"] == 0 and {w["phase"] for w in windows} == {S.NO_EVENT}


# 4 ------------------------------------------------------------------ accident / non-accident mapping (ML)
def test_ml_accident_classes_follow_the_model_not_the_scenario_list(dev_experiment):
    meta = load_summary(dev_experiment)["metadata"]
    model_classes = [str(c) for c in predict_expanded.load_model().classes_]
    assert meta["ml_classes"] == model_classes
    assert meta["ml_accident_classes"] == ["Minor Accident", "Severe Accident"]
    w = pd.read_csv(dev_experiment / "window_results.csv")
    assert (w["ml_binary"] == w["ml_label"].isin(meta["ml_accident_classes"]).astype(int)).all()


# 5 ------------------------------------------------------------------ identical sensor stream
def test_both_detectors_receive_the_identical_windows(monkeypatch):
    seen = {"ml": [], "th": []}
    real_ml, real_th = predict_expanded.predict_window_event, td.detect_window

    def spy_ml(window):
        seen["ml"].append(json.dumps(window, sort_keys=True))
        return real_ml(window)

    def spy_th(window, config=None):
        seen["th"].append(json.dumps(list(window), sort_keys=True))
        return real_th(window, config)

    monkeypatch.setattr(predict_expanded, "predict_window_event", spy_ml)
    monkeypatch.setattr(td, "detect_window", spy_th)
    er.run_recording("Minor Accident", 11400, "t", 30, ["Minor Accident"], td.load_default_config())
    assert len(seen["ml"]) == 21 == len(seen["th"]) and seen["ml"] == seen["th"]


def test_saved_recording_is_the_exact_stream_both_detectors_saw(tmp_path):
    store = RecordingStore(tmp_path)
    row, windows = er.run_recording("Severe Accident", 11500, "t", 50, ["Minor Accident", "Severe Accident"],
                                    td.load_default_config(), store)
    rec = store.load(row["recording_id"])
    assert rec.data["source_type"] == "virtual_hardware" and len(rec.samples) == 50
    assert rec.data["virtual_hardware"]["event_onset_tick"] == row["event_onset_tick"]
    src = VirtualHardwareSensorSource(seed=11500, run_ticks=50)
    src.start("Severe Accident")
    fresh = [src.read() for _ in range(50)]
    for s, f in zip(rec.samples, fresh):
        assert all(s[k] == pytest.approx(f[k]) for k in FEATURE_COLUMNS)
    # re-running the frozen threshold detector on the SAVED samples reproduces the recorded decisions
    thr = td.load_default_config()
    for w in windows:
        i = w["window_end_tick"]
        win = [{k: rec.samples[j][k] for k in FEATURE_COLUMNS} for j in range(i - WINDOW_SIZE + 1, i + 1)]
        assert td.detect_window(win, thr)["event_detected"] == bool(w["threshold_binary"])
    assert len(rec.decisions) == len(windows)
    assert rp.ReplaySession(rec).n_samples == 50                            # Phase 10 replay works on it


# 6-7 --------------------------------------------------------------- the two methods
def test_ml_window_records(dev_experiment):
    w = pd.read_csv(dev_experiment / "window_results.csv")
    probs = [c for c in w.columns if c.startswith("ml_prob_")]
    assert sorted(c[len("ml_prob_"):] for c in probs) == sorted(str(c) for c in predict_expanded.load_model().classes_)
    assert (w[probs].sum(axis=1) - 1).abs().max() < 1e-6
    assert (w["ml_confidence"] == w[probs].max(axis=1)).all() or (w["ml_confidence"] <= 1).all()
    assert set(w["ml_label"]) <= {c[len("ml_prob_"):] for c in probs}            # the actual class output is preserved


def test_threshold_window_records(dev_experiment):
    w = pd.read_csv(dev_experiment / "window_results.csv").fillna("")
    assert set(w["threshold_binary"]) <= {0, 1}
    assert (w[w["threshold_binary"] == 1]["threshold_rules"] != "").all() and (w[w["threshold_binary"] == 0]["threshold_rules"] == "").all()
    assert set(w["threshold_event_class"]) <= set(td.EVENT_CLASSES)           # secondary cascade preserved


# 8-9 --------------------------------------------------------------- pre-event false alarms and latency
def test_pre_event_false_alarm_is_not_a_detection_and_has_no_negative_latency():
    dec = lambda ticks: [{"end_tick": t, "positive": t in ticks} for t in range(9, 30)]          # noqa: E731
    o = ev.detection_outcome(dec({12, 20}), onset_tick=15, is_accident=True)
    assert o["pre_event_false_alarm"] and o["binary_decision"] == 1                   # both facts recorded
    assert o["detection_tick"] == 20 and o["latency_ms"] == 500 and o["detection_ms"] == 2000
    only_early = ev.detection_outcome(dec({12}), onset_tick=15, is_accident=True)
    assert only_early["pre_event_false_alarm"] and only_early["binary_decision"] == 0
    assert only_early["latency_ms"] is None and only_early["detection_tick"] is None
    clean = ev.detection_outcome(dec(set()), onset_tick=15, is_accident=True)
    assert clean["binary_decision"] == 0 and not clean["pre_event_false_alarm"]


def test_latency_is_simulation_time_from_onset():
    o = ev.detection_outcome([{"end_tick": 15, "positive": True}], onset_tick=15, is_accident=True)
    assert o["latency_ms"] == 0
    o = ev.detection_outcome([{"end_tick": 18, "positive": True}], onset_tick=15, is_accident=True)
    assert o["latency_ms"] == 300 and o["detection_ms"] == 1800
    n = ev.detection_outcome([{"end_tick": 18, "positive": True}], None, is_accident=False)
    assert n["binary_decision"] == 1 and n["latency_ms"] is None and not n["pre_event_false_alarm"]
    with pytest.raises(ValueError):
        ev.detection_outcome([], None, is_accident=True)


def test_latency_in_results_matches_vehicle_clock(dev_experiment):
    r = pd.read_csv(dev_experiment / "recordings.csv")
    acc = r[(r["ground_truth"] == 1) & (r["ml_binary_decision"] == 1)]
    assert (acc["ml_latency_ms"] == acc["ml_detection_ms"] - acc["event_onset_ms"]).all() and (acc["ml_latency_ms"] >= 0).all()
    assert (r["event_onset_ms"].dropna() % 100 == 0).all()
    # tick 0 = 0 ms, matching the first telemetry timestamp minus one period
    src = VirtualHardwareSensorSource(seed=11000, run_ticks=3)
    src.start("Normal Driving")
    src.read()
    assert src.last_timestamp_ms - 100 == 0


# 10-15 ------------------------------------------------------------- binary metrics
def test_confusion_counts_and_derived_metrics():
    truth = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    c = ev.confusion_counts(truth, pred)
    assert c == {"TP": 3, "TN": 5, "FP": 1, "FN": 1}                                    # TP/TN/FP/FN
    m = ev.binary_metrics(c)
    assert m["accuracy"] == pytest.approx(0.8) and m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75) and m["f1"] == pytest.approx(0.75)
    assert m["false_positive_rate"] == pytest.approx(1 / 6) and m["false_negative_rate"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        ev.confusion_counts([1], [1, 0])


def test_metrics_with_empty_denominators_are_none_not_zero():
    m = ev.binary_metrics({"TP": 0, "TN": 4, "FP": 0, "FN": 0})
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None and m["false_negative_rate"] is None
    assert m["accuracy"] == 1.0 and m["false_positive_rate"] == 0.0


def test_wilson_interval_and_latency_stats_and_mcnemar():
    lo, hi = ev.wilson_interval(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95 and ev.wilson_interval(0, 0) is None
    s = ev.latency_stats([0, 100, 300, None])
    assert s["n"] == 3 and s["mean"] == pytest.approx(133.33, abs=0.01) and s["median"] == 100 and s["min"] == 0 and s["max"] == 300
    assert s["std"] == pytest.approx(152.75, abs=0.01) and len(s["ci95_mean"]) == 2
    assert ev.latency_stats([None])["n"] == 0
    m = ev.mcnemar_exact([True] * 10 + [False] * 2, [False] * 10 + [True] * 2)
    assert (m["only_first_correct"], m["only_second_correct"], m["discordant"]) == (10, 2, 12)
    assert m["p_value"] == pytest.approx(2 * sum(math.comb(12, i) for i in range(3)) / 2 ** 12)
    assert ev.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


# 16-17 ------------------------------------------------------------- recording vs window level
def test_recording_level_counts_detections_only_after_onset():
    recs = [rec_row(), rec_row(ml_binary_decision=0, ml_latency_ms=None, ml_pre_event_false_alarm=True),
            rec_row(scenario="Pothole", ground_truth=0, ml_binary_decision=1, ml_latency_ms=None, threshold_binary_decision=0,
                    threshold_latency_ms=None),
            rec_row(scenario="Normal Driving", ground_truth=0, ml_binary_decision=0, ml_latency_ms=None,
                    threshold_binary_decision=0, threshold_latency_ms=None)]
    ml = ev.recording_level(recs, "ml")
    assert (ml["TP"], ml["FN"], ml["FP"], ml["TN"]) == (1, 1, 1, 1) and ml["pre_event_false_alarm_count"] == 1
    assert ml["missed_event_count"] == 1 and ml["latency_ms"]["n"] == 1
    th = ev.recording_level(recs, "threshold")
    assert (th["TP"], th["FN"], th["FP"], th["TN"]) == (2, 0, 0, 2) and th["pre_event_false_alarm_count"] == 0


def test_window_level_truth_and_exclusions():
    assert ev.window_truth("Pothole", S.EVENT) == 0 and ev.window_truth("Normal Driving", S.NO_EVENT) == 0
    assert ev.window_truth("Rollover", S.EVENT) == 1 and ev.window_truth("Rollover", S.PRE_EVENT) == 0
    assert ev.window_truth("Rollover", S.POST_EVENT) is None
    w = lambda sc, ph, p: {"scenario": sc, "phase": ph, "ml_binary": p}                 # noqa: E731
    rows = [w("Rollover", S.PRE_EVENT, 1), w("Rollover", S.EVENT, 1), w("Rollover", S.EVENT, 0),
            w("Rollover", S.POST_EVENT, 1), w("Pothole", S.EVENT, 0), w("Pothole", S.EVENT, 1)]
    m = ev.window_level(rows, "ml_binary")
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 2, 1)
    assert m["excluded_post_event_windows"] == 1 and m["pre_event_false_alarm_windows"] == 1


# 18 ---------------------------------------------------------------- out-of-distribution handling
def test_rollover_and_multi_impact_are_never_scored_as_ml_classes(dev_experiment):
    summary = load_summary(dev_experiment)
    s, meta = summary["summary"], summary["metadata"]
    assert meta["ml_out_of_distribution_scenarios"] == ["Rollover", "Multi-Impact Collision"]
    mc = s["ml_multiclass"]
    assert "Rollover" not in mc["classes"] and "Multi-Impact Collision" not in mc["classes"] and len(mc["classes"]) == 6
    assert set(s["ml_out_of_distribution_predictions"]) == {"Rollover", "Multi-Impact Collision"}
    for dist in s["ml_out_of_distribution_predictions"].values():
        assert set(dist) <= set(mc["classes"])                       # actual predicted classes, all from the model
    rows = {r["scenario"]: r for r in s["per_scenario"]}
    assert rows["Rollover"]["ml_out_of_distribution"] and rows["Multi-Impact Collision"]["ml_out_of_distribution"]
    assert not rows["Severe Accident"]["ml_out_of_distribution"]
    assert all(v["support"] >= 0 for v in mc["per_class"].values())
    classes = ["Normal Driving", "Pothole"]
    win = [{"scenario": "Rollover", "phase": S.EVENT, "ml_label": "Pothole"},
           {"scenario": "Pothole", "phase": S.EVENT, "ml_label": "Pothole"},
           {"scenario": "Pothole", "phase": S.PRE_EVENT, "ml_label": "Normal Driving"}]
    m = ev.multiclass_confusion(win, classes)
    assert m["matrix"] == [[0, 0], [0, 1]] and m["per_class"]["Pothole"]["support"] == 1
    assert ev.ood_distribution(win, classes) == {"Rollover": {"Pothole": 1}}


# 19-21 ------------------------------------------------------------- result files
def test_result_files_exist_with_expected_schema(dev_experiment):
    for name in ("config.json", "phase11_results.json", "phase11_summary.json", "summary.json", "phase11_results.csv",
                 "recordings.csv", "window_results.csv", "phase11_per_scenario.csv", "per_scenario.csv",
                 "phase11_confusion_matrix.csv", "confusion_ml.csv", "confusion_threshold.csv", "confusion_ml_multiclass.csv"):
        assert (dev_experiment / name).is_file(), name
    results = json.loads((dev_experiment / "phase11_results.json").read_text())
    assert set(results) == {"metadata", "summary", "recordings"} and len(results["recordings"]) == 8 * DEV_RUNS
    s = results["summary"]
    assert set(s) >= {"counts", "recording_level", "window_level", "per_scenario", "ml_multiclass", "paired_comparison",
                      "ml_out_of_distribution_predictions", "saturation_effect", "definitions"}
    for m in S.METHODS:
        assert set(s["recording_level"][m]) >= {"TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "f1",
                                                "false_positive_rate", "false_negative_rate", "latency_ms",
                                                "pre_event_false_alarm_count", "missed_event_count"}
        assert set(s["recording_level"][m]["latency_ms"]) == {"n", "mean", "median", "std", "min", "max", "ci95_mean"}
    assert s["paired_comparison"]["all_recordings"]["n_pairs"] == 8 * DEV_RUNS
    assert "p_value" in s["paired_comparison"]["accident_recordings_detected"]


def test_csv_schema_counts_and_uniqueness(dev_experiment):
    r = pd.read_csv(dev_experiment / "recordings.csv")
    assert list(r.columns) == S.RECORDING_COLUMNS and len(r) == 8 * DEV_RUNS
    assert r["recording_id"].is_unique and r["seed"].is_unique
    assert (r.groupby("scenario").size() == DEV_RUNS).all() and set(r["scenario"]) == set(S.ALL_SCENARIOS)
    assert (r["sample_count"] == 50).all() and (r["window_count"] == 41).all()
    assert (r["ground_truth"] == r["scenario"].map(S.GROUND_TRUTH)).all()
    assert r[r["ground_truth"] == 0]["event_onset_tick"].isna().sum() >= DEV_RUNS                 # Normal Driving: no event
    assert r[r["ground_truth"] == 1]["event_onset_tick"].notna().all()
    assert (dev_experiment / "phase11_results.csv").read_text() == (dev_experiment / "recordings.csv").read_text()
    w = pd.read_csv(dev_experiment / "window_results.csv")
    assert len(w) == 8 * DEV_RUNS * 41 and set(w["phase"]) <= {S.PRE_EVENT, S.EVENT, S.POST_EVENT, S.NO_EVENT}
    long = pd.read_csv(dev_experiment / "phase11_confusion_matrix.csv")
    assert list(long.columns) == ["method", "level", "kind", "truth", "predicted", "count"]
    store = RecordingStore(dev_experiment / "recordings")
    ids = {x["recording_id"] for x in store.list_recordings()}
    assert ids == set(r["recording_id"]) and all(x["error"] is None for x in store.list_recordings())


def test_reproducibility_metadata(dev_experiment, dev_cfg):
    meta = load_summary(dev_experiment)["metadata"]
    assert meta["experiment_id"] == "p11_dev_fixture" and meta["runs_per_scenario"] == DEV_RUNS
    assert meta["seed_ranges"]["Normal Driving"] == [11000, 11001] and meta["seed_ranges"]["Rollover"] == [11600, 11601]
    assert meta["scenarios"] == list(S.ALL_SCENARIOS) and meta["sample_rate_hz"] == 10
    assert meta["windowing"]["window_size_samples"] == WINDOW_SIZE and meta["windowing"]["step"] == 1
    assert meta["ml_model_file"] == "accident_classifier_model_expanded.joblib"
    assert meta["ml_model_sha256"] == er.sha256(predict_expanded.EXPANDED_MODEL_PATH)
    assert meta["threshold_config_file"] == "threshold_config_v1.json"
    assert meta["threshold_config_sha256"] == er.sha256(td.DEFAULT_CONFIG_PATH)
    assert meta["frozen_files_unchanged"] is True and meta["finished_utc"] and meta["started_utc"]
    assert meta["software"]["python"] and meta["software"]["scikit-learn"]
    assert any("SYNTHETIC CONTROLLED EXPERIMENT" in x for x in meta["limitations"])
    assert json.loads((dev_experiment / "config.json").read_text())["runs_per_scenario"] == DEV_RUNS


# 22 ---------------------------------------------------------------- no mutation of frozen assets
def test_frozen_files_are_unchanged_and_a_change_aborts_the_run(dev_experiment, dev_cfg, tmp_path, monkeypatch):
    meta = load_summary(dev_experiment)["metadata"]
    assert meta["frozen_file_sha256_before"] == meta["frozen_file_sha256_after"] == er.hash_frozen()
    calls = iter([{"ml_model": "a", "threshold_config": "b", "severity_config": "c"},
                  {"ml_model": "CHANGED", "threshold_config": "b", "severity_config": "c"}])
    monkeypatch.setattr(er, "hash_frozen", lambda: next(calls))
    tiny = dict(dev_cfg, runs_per_scenario=1, scenarios=["Normal Driving"])
    with pytest.raises(RuntimeError, match="FROZEN FILE CHANGED"):
        er.run_experiment(tiny, tmp_path, experiment_id="should_abort")


def test_experiments_are_never_overwritten(dev_experiment, dev_cfg):
    with pytest.raises(FileExistsError):
        er.run_experiment(dev_cfg, dev_experiment.parent, experiment_id=dev_experiment.name)


def test_config_validation(tmp_path):
    cfg = er.load_config()
    for bad in (dict(cfg, scenarios=["Teleport"]), dict(cfg, runs_per_scenario=101), dict(cfg, sample_rate_hz=20)):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            er.load_config(p)


def test_per_scenario_and_paired_and_saturation_tables(dev_experiment):
    s = load_summary(dev_experiment)["summary"]
    assert [r["scenario"] for r in s["per_scenario"]] == list(S.ALL_SCENARIOS)
    for row in s["per_scenario"]:
        for m in S.METHODS:
            key = "detection_rate" if row["ground_truth"] else "false_alarm_rate"
            assert 0 <= row[f"{m}_{key}"] <= 1
    assert set(s["saturation_effect"]) == {"saturated_during_event", "not_saturated_during_event"}
    pc = s["paired_comparison"]["all_recordings"]
    assert pc["discordant"] == pc["only_first_correct"] + pc["only_second_correct"] and 0 <= pc["p_value"] <= 1
