"""
PHASE 11 -- controlled experiment runner.

    python -m experiments.experiment_runner                       # config: experiments/experiment_config.json
    python -m experiments.experiment_runner --runs-per-scenario 3 --out experiments/results_dev

For every (scenario, seed) it generates ONE recording through the full virtual-hardware path

    VirtualVehicle -> MPU6050 / vibration / GPS -> I2C -> virtual STM32 -> UART -> VirtualHardwareSensorSource

and feeds the SAME readings (the same sliding 10-sample windows) to BOTH frozen detectors:

    ML          predict_expanded.predict_window_event   (accident_classifier_model_expanded.joblib, unchanged)
    THRESHOLD   threshold_detector.detect_window        (threshold_config_v1.json, unchanged)

THIS IS A MEASUREMENT TOOL. Nothing is trained, tuned, calibrated, filtered or re-seeded after results are
seen. The model, the threshold config and the severity config are hashed before and after the run and the run
aborts if any of them changed. Ground truth and event onset come from the simulation (see schemas.py), never
from a detector. Time is simulation time (tick * 100 ms), never wall-clock.

All data is SYNTHETIC; results describe the virtual sensor environment, not real-world accident detection.
"""

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from common import FEATURE_COLUMNS
import predict_expanded
import severity_v2
import threshold_detector as td
from experiments import evaluation as ev
from experiments import schemas as S
from realtime_pipeline import RollingWindowBuffer
from recording_store import RecordingStore, SOFTWARE_VERSION, describe_models
from virtual_hardware import VirtualHardwareSensorSource
from window_features import WINDOW_SIZE, STRIDE

CONFIG_PATH = Path(__file__).with_name("experiment_config.json")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")


# ---------------------------------------------------------------------------- config / seeds / hashing
def load_config(path=CONFIG_PATH) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("experiment_name", "runs_per_scenario", "seed_start", "seed_stride_per_scenario", "sample_rate_hz",
                "recording_duration_seconds", "scenarios", "methods"):
        if key not in cfg:
            raise ValueError(f"experiment config is missing '{key}'")
    unknown = [s for s in cfg["scenarios"] if s not in S.GROUND_TRUTH]
    if unknown:
        raise ValueError(f"unknown scenarios in config: {unknown}")
    if cfg["runs_per_scenario"] > cfg["seed_stride_per_scenario"]:
        raise ValueError("runs_per_scenario exceeds seed_stride_per_scenario: seed ranges would overlap")
    if int(cfg["sample_rate_hz"]) != 1000 // S.SAMPLE_PERIOD_MS:
        raise ValueError("the virtual hardware runs at 10 Hz")
    return cfg


def seed_schedule(cfg: Dict[str, Any]) -> List[Tuple[str, int]]:
    """Deterministic (scenario, seed) list; each scenario owns a disjoint seed block."""
    return [(sc, cfg["seed_start"] + i * cfg["seed_stride_per_scenario"] + k)
            for i, sc in enumerate(cfg["scenarios"]) for k in range(cfg["runs_per_scenario"])]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_files() -> Dict[str, str]:
    return {"ml_model": predict_expanded.EXPANDED_MODEL_PATH, "threshold_config": td.DEFAULT_CONFIG_PATH,
            "severity_config": severity_v2.DEFAULT_CONFIG_PATH}


def hash_frozen() -> Dict[str, str]:
    return {k: sha256(p) for k, p in frozen_files().items()}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def recording_id_for(scenario: str, seed: int) -> str:
    return f"p11-{_slug(scenario)}-{seed}"


# ---------------------------------------------------------------------------- one recording
def event_span(shocks: List[float]) -> Tuple[Optional[int], Optional[int]]:
    """(onset_tick, event_end_tick) from the virtual vehicle's own impact envelope; (None, None) = no event."""
    above = [i for i, v in enumerate(shocks) if v > S.ONSET_SHOCK_THRESHOLD]
    return (above[0], above[-1]) if above else (None, None)


def window_phase(end_tick: int, onset: Optional[int], event_end: Optional[int]) -> str:
    if onset is None:
        return S.NO_EVENT
    if end_tick < onset:
        return S.PRE_EVENT
    start = end_tick - WINDOW_SIZE + 1
    return S.POST_EVENT if start > event_end else S.EVENT


def run_recording(scenario: str, seed: int, experiment_id: str, n_samples: int, ml_accident_classes,
                  threshold_cfg: td.ThresholdConfig, store: Optional[RecordingStore] = None,
                  models_info: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate ONE recording and score both detectors on it. Returns (recording_row, window_rows)."""
    rid = recording_id_for(scenario, seed)
    src = VirtualHardwareSensorSource(seed=seed, run_ticks=n_samples)
    src.start(scenario)

    readings, shocks, saturated, gps_pos, hw = [], [], [], [], []
    while True:
        r = src.read()
        if r is None:
            break
        veh = src.last_tick_result.vehicle
        if veh.tick - 1 != len(readings):
            raise RuntimeError("telemetry sample went missing: tick/sample index mismatch")
        readings.append(r)
        shocks.append(float(veh.shock))
        info = src.hardware_info()
        saturated.append(bool(info["saturated"]))
        gps_pos.append(src.gps_position)
        hw.append(info)
    if len(readings) != n_samples:
        raise RuntimeError(f"expected {n_samples} samples, got {len(readings)}")

    onset, event_end = event_span(shocks)                      # ground truth: from the vehicle state only
    truth = S.GROUND_TRUTH[scenario]
    if truth == 1 and onset is None:
        raise RuntimeError(f"accident scenario {scenario!r} produced no event (seed {seed})")

    rec = None
    if store is not None:
        rec = store.start_recording(scenario, "virtual_hardware", float(1000 // S.SAMPLE_PERIOD_MS),
                                    models=models_info, recording_id=rid)
        rec.set_virtual_hardware({
            "experiment_id": experiment_id, "seed": seed, "scenario": scenario, "ground_truth_accident": bool(truth),
            "event_onset_tick": onset, "event_onset_ms": None if onset is None else onset * S.SAMPLE_PERIOD_MS,
            "event_end_tick": event_end, "onset_definition": f"vehicle shock > {S.ONSET_SHOCK_THRESHOLD}",
            "notice": "PHASE 11 SYNTHETIC CONTROLLED EXPERIMENT RECORDING - virtual hardware simulation"})

    windows: List[Dict[str, Any]] = []
    buf = RollingWindowBuffer()
    for i, reading in enumerate(readings):
        buf.push(reading)
        if rec is not None:
            rec.append_sample(reading, gps=gps_pos[i], hardware=hw[i])
        if not buf.is_full():
            continue
        window = buf.as_list()
        label, conf, probs = predict_expanded.predict_window_event(window)       # frozen ML
        th = td.detect_window(window, threshold_cfg)                              # frozen thresholds
        row = {"experiment_id": experiment_id, "recording_id": rid, "scenario": scenario, "seed": seed,
               "window_end_tick": i, "window_end_ms": i * S.SAMPLE_PERIOD_MS,
               "phase": window_phase(i, onset, event_end),
               "ml_label": str(label), "ml_confidence": float(conf),
               "ml_binary": int(str(label) in ml_accident_classes),
               "threshold_binary": int(th["event_detected"]), "threshold_rules": ";".join(th["triggered_rules"]),
               "threshold_event_class": th["event_class"]}
        row.update({f"ml_prob_{k}": float(v) for k, v in probs.items()})
        windows.append(row)
        if rec is not None:
            rec.append_decision(ai={"label": str(label), "confidence": float(conf), "probabilities": probs},
                                threshold=th)
    if rec is not None:
        rec.finish(save=True)

    out: Dict[str, Any] = {
        "experiment_id": experiment_id, "recording_id": rid, "scenario": scenario, "seed": seed,
        "ground_truth": truth, "event_onset_tick": onset,
        "event_onset_ms": None if onset is None else onset * S.SAMPLE_PERIOD_MS, "event_end_tick": event_end,
        "sample_count": len(readings), "window_count": len(windows), "saturated_ticks": sum(saturated),
        "saturated_in_event": 0 if onset is None else sum(saturated[onset:event_end + 1]),
    }
    for m, col in (("ml", "ml_binary"), ("threshold", "threshold_binary")):
        o = ev.detection_outcome([{"end_tick": w["window_end_tick"], "positive": bool(w[col])} for w in windows],
                                 onset, bool(truth))
        out[f"{m}_binary_decision"] = o["binary_decision"]
        out[f"{m}_detection_tick"], out[f"{m}_detection_ms"] = o["detection_tick"], o["detection_ms"]
        out[f"{m}_latency_ms"], out[f"{m}_pre_event_false_alarm"] = o["latency_ms"], o["pre_event_false_alarm"]
        out[f"{m}_positive_windows"] = o["positive_windows"]
    ev_windows = [w for w in windows if w["phase"] in (S.EVENT, S.NO_EVENT)] or windows
    top = max(ev_windows, key=lambda w: w["ml_confidence"])
    out["ml_top_class"], out["ml_max_confidence"] = top["ml_label"], top["ml_confidence"]
    return out, windows


# ---------------------------------------------------------------------------- whole experiment
def run_experiment(cfg: Dict[str, Any], out_root=DEFAULT_RESULTS_DIR, progress: Optional[Callable[[int, int], None]] = None,
                   experiment_id: Optional[str] = None, now: Optional[datetime] = None, verbose: bool = False) -> Path:
    from experiments import reports                      # local import: reports imports this module's constants

    now = now or datetime.now(timezone.utc)
    experiment_id = experiment_id or f"{cfg['experiment_name']}_{now:%Y%m%d_%H%M%S}"
    out_dir = Path(out_root) / experiment_id
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; experiments are never overwritten")

    before = hash_frozen()
    model = predict_expanded.load_model()
    ml_classes = [str(c) for c in model.classes_]
    ml_accident = sorted(set(ml_classes) & set(S.ACCIDENT_SCENARIOS))
    threshold_cfg = td.load_default_config()
    n_samples = int(cfg["recording_duration_seconds"] * cfg["sample_rate_hz"])
    schedule = seed_schedule(cfg)

    meta = reports.build_metadata(cfg, experiment_id, now, before, ml_classes, ml_accident, n_samples, schedule)
    if verbose:
        print(json.dumps(meta, indent=2))

    out_dir.mkdir(parents=True)
    store = RecordingStore(out_dir / "recordings") if cfg.get("save_recordings", True) else None
    models_info = describe_models(predict_expanded.EXPANDED_MODEL_PATH, td.DEFAULT_CONFIG_PATH, severity_v2.DEFAULT_CONFIG_PATH)

    recs, windows = [], []
    for k, (scenario, seed) in enumerate(schedule):
        r, w = run_recording(scenario, seed, experiment_id, n_samples, ml_accident, threshold_cfg, store, models_info)
        recs.append(r)
        windows.extend(w)
        if progress:
            progress(k + 1, len(schedule))

    after = hash_frozen()
    if after != before:
        raise RuntimeError(f"FROZEN FILE CHANGED DURING THE EXPERIMENT: {before} -> {after}")
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["frozen_files_unchanged"] = True
    meta["frozen_file_sha256_after"] = after
    reports.write_reports(out_dir, cfg, meta, recs, windows, ml_classes)
    if verbose:
        print(f"experiment {experiment_id} finished: {len(recs)} recordings, {len(windows)} windows -> {out_dir}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="PHASE 11 controlled experiment runner (frozen ML vs frozen threshold).")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--runs-per-scenario", type=int, default=None, help="override (development runs only)")
    ap.add_argument("--out", default=str(DEFAULT_RESULTS_DIR))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.runs_per_scenario is not None:
        cfg["runs_per_scenario"] = args.runs_per_scenario
        cfg["experiment_name"] += f"_dev{args.runs_per_scenario}"
    print("PHASE 11 - SYNTHETIC CONTROLLED EXPERIMENT (frozen ML model, frozen thresholds; nothing is tuned)")
    run_experiment(cfg, args.out, progress=lambda k, n: print(f"\r{k}/{n} recordings", end="", file=sys.stderr),
                   verbose=True)
    print(file=sys.stderr)


if __name__ == "__main__":
    main()
