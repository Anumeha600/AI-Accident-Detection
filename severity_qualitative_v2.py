"""
PHASE 9 -- Qualitative evaluation of severity_v2 on representative windows.
---------------------------------------------------------------------------
There is NO ground-truth severity label in the dataset, so this script does
NOT compute a severity accuracy and none should be inferred from it. It only
shows what the estimator outputs for freshly simulated recordings, so a human
can judge whether the behaviour is plausible.

Data: 30 NEW synthetic recordings per scenario from seed 9009 -- not part of
expanded_synthetic_dataset_v2.csv, and in particular not the held-out Phase 7
test recordings. Scenario names are used only to group the printout; they are
never passed to severity_v2. For each recording the "peak window" is the one
with the highest severity score (the estimator's own output, not a label).

Output: severity_qualitative_v2.json (+ console summary).

ALL DATA IS SIMULATED/SYNTHETIC. The severity index is a research index, not
a certified scale, and agreement with intuition here is not validation.
"""

import json
from collections import Counter
from typing import Any, Dict

import numpy as np

import dataset_generator_v2 as g2
import severity_v2 as sv
from window_features import extract_window_features, make_windows

QUALITATIVE_SEED = 9009
RECORDINGS_PER_SCENARIO = 30
OUTPUT_JSON = "severity_qualitative_v2.json"


def evaluate(config: sv.SeverityConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(QUALITATIVE_SEED)
    summary: Dict[str, Any] = {}
    examples: Dict[str, Any] = {}
    for scenario in g2.SCENARIOS_V2:
        peak_levels, peak_scores, all_levels = [], [], []
        best = None
        for rid in range(RECORDINGS_PER_SCENARIO):
            rec = g2.generate_recording(rng, scenario, rid)
            results = [sv.assess(extract_window_features(w), config) for w in make_windows(rec)]
            all_levels += [r["severity"] for r in results]
            top = max(results, key=lambda r: r["score"])
            peak_levels.append(top["severity"])
            peak_scores.append(top["score"])
            if best is None or top["score"] > best["score"]:
                best = top
        order = sv.SEVERITY_LEVELS
        summary[scenario] = {
            "n_recordings": RECORDINGS_PER_SCENARIO,
            "peak_window_level_counts": {l: Counter(peak_levels).get(l, 0) for l in order},
            "peak_window_score": {"min": float(np.min(peak_scores)), "median": float(np.median(peak_scores)),
                                  "max": float(np.max(peak_scores))},
            "all_window_level_counts": {l: Counter(all_levels).get(l, 0) for l in order},
        }
        median_idx = int(np.argsort(peak_scores)[len(peak_scores) // 2])
        examples[scenario] = {"median_peak_window_score": float(peak_scores[median_idx]),
                              "note": "see 'highest_scoring_example' for a full explanation"}
        examples[scenario]["highest_scoring_example"] = best
    return {
        "disclaimer": ("SIMULATED data; severity is a research index, not a certified scale. No severity accuracy "
                       "is computed because the dataset has no ground-truth severity labels; formal validation "
                       "would need new, independently labelled severity data."),
        "data": f"{RECORDINGS_PER_SCENARIO} fresh recordings/scenario, seed {QUALITATIVE_SEED} (not in the dataset)",
        "per_scenario": summary,
        "examples": examples,
    }


def main() -> None:
    cfg = sv.load_default_config()
    report = evaluate(cfg)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("=" * 78)
    print("PHASE 9: severity_v2 qualitative check (fresh SYNTHETIC recordings; no accuracy claimed)")
    print("=" * 78)
    print(f"{'scenario (grouping only)':26s}{'LOW':>5}{'MED':>5}{'HIGH':>6}{'CRIT':>6}   peak-window score min/median/max")
    for sc, s in report["per_scenario"].items():
        c, p = s["peak_window_level_counts"], s["peak_window_score"]
        print(f"{sc:26s}{c['LOW']:>5}{c['MEDIUM']:>5}{c['HIGH']:>6}{c['CRITICAL']:>6}   "
              f"{p['min']:.1f} / {p['median']:.1f} / {p['max']:.1f}")
    print("\n(counts = each recording's highest-severity window; 30 recordings per scenario)")
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
