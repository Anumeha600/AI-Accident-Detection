"""
PHASE 11 -- metadata, summary assembly and result files.

Files written into experiments/results/<experiment_id>/ (a directory is never overwritten):

    config.json                    the exact configuration that was run
    phase11_results.json           everything: metadata + summary + one row per recording
    phase11_summary.json           metadata + summary (also saved as summary.json)
    phase11_results.csv            one row per recording (also saved as recordings.csv)
    window_results.csv             one row per evaluated window (both detectors side by side)
    phase11_per_scenario.csv       per-scenario table (also saved as per_scenario.csv)
    phase11_confusion_matrix.csv   all confusion matrices in long format
    confusion_ml.csv / confusion_threshold.csv      recording-level binary matrices
    confusion_ml_multiclass.csv    ML window-level matrix over the classes the model really has
    recordings/                    the actual sensor recordings (Phase 10 format) so every run can be replayed
"""

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import sklearn

from experiments import evaluation as ev
from experiments import schemas as S
from recording_store import SOFTWARE_VERSION
from window_features import WINDOW_SIZE, STRIDE
import predict_expanded
import threshold_detector as td

LIMITATIONS = [
    "SYNTHETIC CONTROLLED EXPERIMENT: all sensor data come from the virtual-hardware simulator.",
    "Results measure performance within the virtual sensor environment and do not establish real-world "
    "accident-detection accuracy.",
    "The existing ML model is frozen and was not retrained, tuned or re-calibrated during this experiment; "
    "likewise the threshold configuration.",
    "The ML model was trained on unclipped synthetic signals; the virtual MPU6050 clips to +-2 g / +-250 deg/s, "
    "so a train/live distribution mismatch is part of what is being measured.",
    "Rollover and Multi-Impact Collision are not classes of the ML model (out-of-distribution): its class output "
    "for them is reported, never scored as a class, and only enters the binary accident/non-accident metrics.",
    "Finite number of recordings; recordings of one scenario come from one parametric generator and are not "
    "independent samples of any real population. Confidence intervals and p-values describe this simulator only.",
    "Recordings last 5 s (50 samples); a late Rollover / Multi-Impact event can be cut off by the end of the recording.",
]


def build_metadata(cfg, experiment_id, now: datetime, hashes, ml_classes, ml_accident, n_samples, schedule) -> Dict[str, Any]:
    seeds: Dict[str, List[int]] = {}
    for sc, seed in schedule:
        seeds.setdefault(sc, []).append(seed)
    return {
        "experiment_id": experiment_id, "experiment_name": cfg["experiment_name"],
        "started_utc": now.isoformat(), "finished_utc": None,
        "scenarios": list(cfg["scenarios"]), "runs_per_scenario": cfg["runs_per_scenario"],
        "n_recordings_planned": len(schedule),
        "seed_ranges": {sc: [min(v), max(v)] for sc, v in seeds.items()},
        "sample_rate_hz": cfg["sample_rate_hz"], "recording_duration_s": cfg["recording_duration_seconds"],
        "samples_per_recording": n_samples,
        "windowing": {"window_size_samples": WINDOW_SIZE, "window_seconds": WINDOW_SIZE / cfg["sample_rate_hz"],
                      "step": 1, "note": ("sliding windows with stride 1 (what the live app does); both detectors receive "
                                          "the identical window list. Both compute the SAME 36 window features with "
                                          "window_features.extract_window_features (the threshold detector uses a subset "
                                          f"of them); the offline training stride was {STRIDE}.")},
        "ml_model_file": Path(predict_expanded.EXPANDED_MODEL_PATH).name,
        "ml_model_sha256": hashes["ml_model"], "ml_classes": ml_classes, "ml_accident_classes": ml_accident,
        "ml_out_of_distribution_scenarios": [s for s in S.ACCIDENT_SCENARIOS if s not in ml_classes],
        "threshold_config_file": Path(td.DEFAULT_CONFIG_PATH).name, "threshold_config_sha256": hashes["threshold_config"],
        "frozen_file_sha256_before": hashes, "frozen_files_unchanged": None,
        "ground_truth": {"accident_scenarios": list(S.ACCIDENT_SCENARIOS),
                         "non_accident_scenarios": list(S.NON_ACCIDENT_SCENARIOS),
                         "source": "simulation scenario + virtual vehicle state (never a detector)",
                         "event_onset_definition": f"first sample with vehicle shock > {S.ONSET_SHOCK_THRESHOLD}",
                         "time_base": "simulation time, tick * 100 ms (not wall-clock)"},
        "software": {"version": SOFTWARE_VERSION, "python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "limitations": LIMITATIONS,
    }


def build_summary(recs: Sequence[Dict[str, Any]], windows: Sequence[Dict[str, Any]], ml_classes: Sequence[str]) -> Dict[str, Any]:
    recording = {m: ev.recording_level(recs, m) for m in S.METHODS}
    window = {"ml": ev.window_level(windows, "ml_binary"), "threshold": ev.window_level(windows, "threshold_binary")}
    mc = ev.multiclass_confusion(windows, ml_classes)
    return {
        "counts": {"recordings": len(recs), "windows": len(windows),
                   "per_scenario": {sc: sum(r["scenario"] == sc for r in recs) for sc in S.ALL_SCENARIOS
                                    if any(r["scenario"] == sc for r in recs)},
                   "unique_recording_ids": len({r["recording_id"] for r in recs})},
        "recording_level": recording, "window_level": window,
        "per_scenario": ev.per_scenario(recs, windows, ml_classes),
        "ml_multiclass": mc,
        "ml_out_of_distribution_predictions": ev.ood_distribution(windows, ml_classes),
        "paired_comparison": ev.paired_comparison(recs),
        "saturation_effect": ev.saturation_effect(recs),
        "definitions": {
            "recording_level": ("accident recording DETECTED iff the detector fires on a window ending at or after the true "
                                "onset; non-accident recording FALSE ALARM iff it fires on any window; positives before the "
                                "onset of an accident recording are PRE_EVENT_FALSE_ALARMs (never detections, no latency)."),
            "window_level": ("each window scored independently; accident-recording windows count as positive only in phase "
                             "EVENT, PRE_EVENT windows are negatives, POST_EVENT windows are excluded; all windows of "
                             "non-accident recordings are negatives."),
            "latency": "(first valid positive window end tick - onset tick) * 100 ms, simulation time",
        },
    }


def _confusion_frame(c: Dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame([[c["TN"], c["FP"]], [c["FN"], c["TP"]]], index=["truth_non_accident", "truth_accident"],
                        columns=["pred_non_accident", "pred_accident"])


def write_reports(out_dir: Path, cfg, meta, recs, windows, ml_classes) -> None:
    out_dir = Path(out_dir)
    summary = build_summary(recs, windows, ml_classes)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    head = {"metadata": meta, "summary": summary}
    for name in ("phase11_summary.json", "summary.json"):
        (out_dir / name).write_text(json.dumps(head, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "phase11_results.json").write_text(
        json.dumps({**head, "recordings": list(recs)}, indent=1, allow_nan=False), encoding="utf-8")

    rec_df = pd.DataFrame(recs)[S.RECORDING_COLUMNS]
    for name in ("phase11_results.csv", "recordings.csv"):
        rec_df.to_csv(out_dir / name, index=False)
    pd.DataFrame(windows).to_csv(out_dir / "window_results.csv", index=False)
    ps = pd.DataFrame(summary["per_scenario"])
    for name in ("phase11_per_scenario.csv", "per_scenario.csv"):
        ps.to_csv(out_dir / name, index=False)

    long_rows = []
    for m in S.METHODS:
        frame = _confusion_frame(summary["recording_level"][m])
        frame.to_csv(out_dir / f"confusion_{m}.csv")
        for level, c in (("recording", summary["recording_level"][m]), ("window", summary["window_level"][m])):
            for t, p, key in (("non_accident", "non_accident", "TN"), ("non_accident", "accident", "FP"),
                              ("accident", "non_accident", "FN"), ("accident", "accident", "TP")):
                long_rows.append({"method": m, "level": level, "kind": "binary", "truth": t, "predicted": p, "count": c[key]})
    mc = summary["ml_multiclass"]
    pd.DataFrame(mc["matrix"], index=[f"truth_{c}" for c in mc["classes"]],
                 columns=[f"pred_{c}" for c in mc["classes"]]).to_csv(out_dir / "confusion_ml_multiclass.csv")
    for i, t in enumerate(mc["classes"]):
        for j, p in enumerate(mc["classes"]):
            long_rows.append({"method": "ml", "level": "window", "kind": "multiclass", "truth": t, "predicted": p,
                              "count": mc["matrix"][i][j]})
    pd.DataFrame(long_rows).to_csv(out_dir / "phase11_confusion_matrix.csv", index=False)
