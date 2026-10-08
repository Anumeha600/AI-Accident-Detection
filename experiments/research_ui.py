"""
RESEARCH LAB tab: shows the results of a Phase 11 controlled experiment, read from the result files the
runner wrote (experiments/results/<experiment_id>/phase11_summary.json). There are NO placeholder numbers:
without result files the tab says so. Nothing runs when the page loads; the experiment only runs when the
button is pressed.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from experiments import schemas as S

RESULTS_DIR_ENV = "ACCIDENT_EXPERIMENT_RESULTS_DIR"
NO_RESULTS = "No experiment results available."


def results_dir() -> Path:
    return Path(os.environ.get(RESULTS_DIR_ENV, Path(__file__).with_name("results")))


def list_experiments(root: Optional[Path] = None) -> List[Path]:
    """Experiment directories that contain a summary, newest first (by started_utc)."""
    root = Path(root) if root is not None else results_dir()
    if not root.is_dir():
        return []
    found = []
    for d in root.iterdir():
        f = d / "phase11_summary.json"
        if f.is_file():
            try:
                found.append((json.loads(f.read_text(encoding="utf-8"))["metadata"]["started_utc"], d))
            except (OSError, ValueError, KeyError):
                continue
    return [d for _, d in sorted(found, reverse=True)]


def load_summary(directory: Path) -> Dict[str, Any]:
    return json.loads((Path(directory) / "phase11_summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- tables (pure; tested)
def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.1f}" if isinstance(v, float) else str(v)


def metrics_table(summary: Dict[str, Any], level: str = "recording_level") -> pd.DataFrame:
    rows = [("TP", "TP", False), ("TN", "TN", False), ("FP", "FP", False), ("FN", "FN", False),
            ("Accuracy", "accuracy", True), ("Precision", "precision", True), ("Recall", "recall", True),
            ("F1", "f1", True), ("False positive rate", "false_positive_rate", True),
            ("False negative rate", "false_negative_rate", True)]
    src = summary["summary"][level]
    return pd.DataFrame({"Metric": [r[0] for r in rows],
                         "ML": [_fmt(src["ml"][r[1]], r[2]) for r in rows],
                         "Threshold": [_fmt(src["threshold"][r[1]], r[2]) for r in rows]})


def latency_table(summary: Dict[str, Any]) -> pd.DataFrame:
    rec = summary["summary"]["recording_level"]
    rows = [("Detected accident recordings (n)", "n", False), ("Median (ms)", "median", True), ("Mean (ms)", "mean", True),
            ("Std (ms)", "std", True), ("Min (ms)", "min", True), ("Max (ms)", "max", True)]
    out = {"": [r[0] for r in rows]}
    for m, name in (("ml", "ML"), ("threshold", "Threshold")):
        out[name] = [_fmt(rec[m]["latency_ms"][r[1]]) for r in rows]
    out[""].append("Missed events")
    out[""].append("Pre-event false alarms")
    for m, name in (("ml", "ML"), ("threshold", "Threshold")):
        out[name] += [str(rec[m]["missed_event_count"]), str(rec[m]["pre_event_false_alarm_count"])]
    return pd.DataFrame(out)


def scenario_table(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in summary["summary"]["per_scenario"]:
        acc = r["ground_truth"] == 1
        key = "detection_rate" if acc else "false_alarm_rate"
        rows.append({"Scenario": r["scenario"] + ("  (ML out-of-distribution)" if r["ml_out_of_distribution"] else ""),
                     "Truth": "accident" if acc else "non-accident", "Recordings": r["n_recordings"],
                     "Rate shown": "detection rate" if acc else "false-alarm rate",
                     "ML": _fmt(r[f"ml_{key}"], True), "Threshold": _fmt(r[f"threshold_{key}"], True),
                     "ML median latency (ms)": _fmt(r["ml_median_latency_ms"]),
                     "Threshold median latency (ms)": _fmt(r["threshold_median_latency_ms"]),
                     "ML pre-event alarms": _fmt(r["ml_pre_event_false_alarm_rate"], True),
                     "Threshold pre-event alarms": _fmt(r["threshold_pre_event_false_alarm_rate"], True)})
    return pd.DataFrame(rows)


def confusion_tables(summary: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    out = {}
    for m, name in (("ml", "ML"), ("threshold", "Threshold")):
        c = summary["summary"]["recording_level"][m]
        out[f"{name} (recording level)"] = pd.DataFrame([[c["TN"], c["FP"]], [c["FN"], c["TP"]]],
                                                        index=["truth: non-accident", "truth: accident"],
                                                        columns=["pred: non-accident", "pred: accident"])
    mc = summary["summary"]["ml_multiclass"]
    out["ML multiclass (window level, model classes only)"] = pd.DataFrame(
        mc["matrix"], index=[f"truth: {c}" for c in mc["classes"]], columns=[f"pred: {c}" for c in mc["classes"]])
    return out


def ood_table(summary: Dict[str, Any]) -> pd.DataFrame:
    dist = summary["summary"]["ml_out_of_distribution_predictions"]
    classes = summary["summary"]["ml_multiclass"]["classes"]
    return pd.DataFrame([{"Scenario (no ML class)": sc, **{c: d.get(c, 0) for c in classes}} for sc, d in dist.items()])


# ---------------------------------------------------------------------------- page
def render(st, run_experiment=None, load_config=None):
    """Draw the RESEARCH LAB tab. `run_experiment` / `load_config` are injected by app.py (kept out of module import)."""
    st.subheader("RESEARCH LAB")
    st.warning("**SYNTHETIC CONTROLLED EXPERIMENT.** Results measure performance within the virtual sensor environment "
               "and do not establish real-world accident-detection accuracy. The existing ML model is frozen and was "
               "not retrained during this experiment; the threshold detector is likewise frozen.")

    with st.expander("Run a controlled experiment", expanded=False):
        if run_experiment is None:
            st.caption("The runner is not available in this session.")
        else:
            cfg = load_config()
            runs = st.number_input("Recordings per scenario", min_value=1, max_value=100,
                                   value=int(cfg["runs_per_scenario"]), key="research_runs")
            total = int(runs) * len(cfg["scenarios"])
            st.caption(f"{total} recordings ({len(cfg['scenarios'])} scenarios x {int(runs)}), fixed seed schedule from "
                       f"{cfg['seed_start']}. A full 400-recording run takes roughly ten minutes; nothing is run "
                       "until you press the button, and finished experiments are never overwritten.")
            if st.button("RUN EXPERIMENT", key="research_run"):
                cfg = dict(cfg, runs_per_scenario=int(runs))
                if int(runs) != 50:
                    cfg["experiment_name"] += f"_{int(runs)}per"
                bar = st.progress(0.0, text="starting...")
                out = run_experiment(cfg, results_dir(), progress=lambda k, n: bar.progress(k / n, text=f"{k}/{n} recordings"))
                st.session_state["research_selected"] = out.name
                st.success(f"Finished: {out.name}")

    experiments = list_experiments()
    if not experiments:
        st.info(NO_RESULTS)
        return
    names = [d.name for d in experiments]
    default = st.session_state.get("research_selected")
    choice = st.selectbox("Experiment", names, index=names.index(default) if default in names else 0)
    summary = load_summary(experiments[names.index(choice)])
    meta, res = summary["metadata"], summary["summary"]
    c = res["counts"]
    st.markdown(f"**{meta['experiment_name']}** · {c['recordings']} recordings ({meta['runs_per_scenario']} per scenario) · "
                f"{c['windows']} windows · seeds {min(r[0] for r in meta['seed_ranges'].values())}"
                f"–{max(r[1] for r in meta['seed_ranges'].values())}")

    st.markdown("#### Results — recording level")
    st.caption(res["definitions"]["recording_level"])
    st.dataframe(metrics_table(summary), hide_index=True, width="stretch")
    with st.expander("Window-level metrics"):
        st.caption(res["definitions"]["window_level"])
        st.dataframe(metrics_table(summary, "window_level"), hide_index=True, width="stretch")

    st.markdown("#### Detection latency (simulation time from true event onset)")
    st.dataframe(latency_table(summary), hide_index=True, width="stretch")

    st.markdown("#### Scenario results")
    st.dataframe(scenario_table(summary), hide_index=True, width="stretch")

    ood = meta["ml_out_of_distribution_scenarios"]
    if ood:
        st.warning(f"Out-of-distribution for the ML model (no such class): {', '.join(ood)}. Its predictions for them "
                   "are shown below, never scored as classes; they only count in the binary accident decision.")
        st.dataframe(ood_table(summary), hide_index=True, width="stretch")

    pc = res["paired_comparison"]["all_recordings"]
    sat = res["saturation_effect"]
    st.markdown("#### Paired comparison and sensor saturation")
    st.caption(f"{res['paired_comparison']['test']}: only ML right {pc['only_first_correct']}, only threshold right "
               f"{pc['only_second_correct']}, p = {pc['p_value']:.3g}. {res['paired_comparison']['caveat']}")
    st.dataframe(pd.DataFrame([{"Accident recordings": k.replace("_", " "), "n": v["n"],
                                "ML detection rate": _fmt(v["ml_detection_rate"], True),
                                "Threshold detection rate": _fmt(v["threshold_detection_rate"], True)}
                               for k, v in sat.items()]), hide_index=True, width="stretch")

    with st.expander("Confusion matrices"):
        for name, df in confusion_tables(summary).items():
            st.markdown(f"**{name}**")
            st.dataframe(df, width="stretch")

    with st.expander("Reproducibility and limitations"):
        st.json({k: meta[k] for k in ("experiment_id", "started_utc", "finished_utc", "seed_ranges", "scenarios",
                                      "runs_per_scenario", "sample_rate_hz", "windowing", "ml_model_file",
                                      "ml_model_sha256", "threshold_config_file", "threshold_config_sha256",
                                      "frozen_files_unchanged", "software", "ground_truth")}, expanded=False)
        for line in meta["limitations"]:
            st.markdown(f"- {line}")
