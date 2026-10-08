"""
PHASE 11 -- pure evaluation functions (no models, no I/O). Inputs are plain dict rows produced by the
runner; every number reported is derived from them.

Two LEVELS are kept apart and never mixed:
    WINDOW level     every labelled window is scored independently (phase POST_EVENT excluded, see schemas)
    RECORDING level  one outcome per recording: an accident recording is DETECTED iff the detector fired on a
                     window that ends AT OR AFTER the true event onset; a non-accident recording is a FALSE
                     ALARM iff the detector fired on any window. A positive BEFORE the onset of an accident
                     recording is a PRE_EVENT_FALSE_ALARM: it is never a detection and has no latency.
"""

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

from experiments import schemas as S


def _div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


# ---------------------------------------------------------------------------- binary metrics
def confusion_counts(truth: Sequence[int], pred: Sequence[int]) -> Dict[str, int]:
    if len(truth) != len(pred):
        raise ValueError("truth and pred differ in length")
    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def binary_metrics(c: Dict[str, int]) -> Dict[str, Any]:
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else \
        2 * precision * recall / (precision + recall)
    return {**c, "n": tp + tn + fp + fn, "accuracy": _div(tp + tn, tp + tn + fp + fn), "precision": precision,
            "recall": recall, "f1": f1, "false_positive_rate": _div(fp, fp + tn),
            "false_negative_rate": _div(fn, fn + tp)}


def wilson_interval(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


# ---------------------------------------------------------------------------- latency
def latency_stats(latencies: Iterable[Optional[float]]) -> Dict[str, Any]:
    v = [float(x) for x in latencies if x is not None]
    if not v:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "ci95_mean": None}
    std = statistics.stdev(v) if len(v) > 1 else None
    ci = None if std is None else [statistics.fmean(v) - 1.96 * std / math.sqrt(len(v)),
                                   statistics.fmean(v) + 1.96 * std / math.sqrt(len(v))]
    return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v), "std": std,
            "min": min(v), "max": max(v), "ci95_mean": ci}


def detection_outcome(decisions: Sequence[Dict[str, Any]], onset_tick: Optional[int], is_accident: bool) -> Dict[str, Any]:
    """
    One method's recording-level outcome. `decisions` = [{"end_tick": int, "positive": bool}, ...] in time order.
    Latency is measured from the true onset to the first positive window ending at or after it, in
    simulation time (ticks * 100 ms).
    """
    positives = [d["end_tick"] for d in decisions if d["positive"]]
    if not is_accident:
        return {"binary_decision": int(bool(positives)), "detection_tick": None, "detection_ms": None,
                "latency_ms": None, "pre_event_false_alarm": False, "positive_windows": len(positives)}
    if onset_tick is None:
        raise ValueError("an accident recording must have an event onset")
    pre = [t for t in positives if t < onset_tick]
    post = [t for t in positives if t >= onset_tick]
    det = post[0] if post else None
    return {"binary_decision": int(det is not None), "detection_tick": det,
            "detection_ms": None if det is None else det * S.SAMPLE_PERIOD_MS,
            "latency_ms": None if det is None else (det - onset_tick) * S.SAMPLE_PERIOD_MS,
            "pre_event_false_alarm": bool(pre), "positive_windows": len(positives)}


# ---------------------------------------------------------------------------- window level
def window_truth(scenario: str, phase: str) -> Optional[int]:
    """Window-level binary truth; None = excluded (POST_EVENT window of an accident recording)."""
    if S.GROUND_TRUTH[scenario] == 0:
        return 0
    if phase == S.EVENT:
        return 1
    if phase == S.PRE_EVENT:
        return 0
    return None


def window_level(windows: Sequence[Dict[str, Any]], method_col: str) -> Dict[str, Any]:
    """`method_col` is the per-window 0/1 decision column (e.g. 'ml_binary')."""
    truth, pred, pre_fp, excluded = [], [], 0, 0
    for w in windows:
        t = window_truth(w["scenario"], w["phase"])
        if t is None:
            excluded += 1
            continue
        truth.append(t)
        pred.append(int(w[method_col]))
        pre_fp += int(w["phase"] == S.PRE_EVENT and S.GROUND_TRUTH[w["scenario"]] == 1 and w[method_col] == 1)
    m = binary_metrics(confusion_counts(truth, pred))
    m.update({"excluded_post_event_windows": excluded, "pre_event_false_alarm_windows": pre_fp})
    return m


# ---------------------------------------------------------------------------- recording level
def recording_level(recs: Sequence[Dict[str, Any]], method: str) -> Dict[str, Any]:
    truth = [r["ground_truth"] for r in recs]
    pred = [int(r[f"{method}_binary_decision"]) for r in recs]
    m = binary_metrics(confusion_counts(truth, pred))
    acc = [r for r in recs if r["ground_truth"] == 1]
    non = [r for r in recs if r["ground_truth"] == 0]
    m["recall_ci95_wilson"] = wilson_interval(m["TP"], m["TP"] + m["FN"])
    m["false_positive_rate_ci95_wilson"] = wilson_interval(m["FP"], m["FP"] + m["TN"])
    m["missed_event_count"] = m["FN"]
    m["pre_event_false_alarm_count"] = sum(bool(r[f"{method}_pre_event_false_alarm"]) for r in acc)
    m["pre_event_false_alarm_rate"] = _div(m["pre_event_false_alarm_count"], len(acc))
    m["latency_ms"] = latency_stats(r[f"{method}_latency_ms"] for r in acc)
    m["n_accident_recordings"], m["n_non_accident_recordings"] = len(acc), len(non)
    return m


def per_scenario(recs: Sequence[Dict[str, Any]], windows: Sequence[Dict[str, Any]], ml_classes: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for sc in S.ALL_SCENARIOS:
        rs = [r for r in recs if r["scenario"] == sc]
        if not rs:
            continue
        ws = [w for w in windows if w["scenario"] == sc]
        row: Dict[str, Any] = {"scenario": sc, "ground_truth": S.GROUND_TRUTH[sc], "n_recordings": len(rs),
                               "ml_out_of_distribution": sc not in ml_classes and S.GROUND_TRUTH[sc] == 1}
        for m in S.METHODS:
            pos = sum(int(r[f"{m}_binary_decision"]) for r in rs)
            lat = latency_stats(r[f"{m}_latency_ms"] for r in rs)
            if S.GROUND_TRUTH[sc] == 1:
                row[f"{m}_detection_rate"] = pos / len(rs)
                row[f"{m}_false_alarm_rate"] = None
                row[f"{m}_pre_event_false_alarm_rate"] = sum(bool(r[f"{m}_pre_event_false_alarm"]) for r in rs) / len(rs)
            else:
                row[f"{m}_detection_rate"] = None
                row[f"{m}_false_alarm_rate"] = pos / len(rs)
                row[f"{m}_pre_event_false_alarm_rate"] = None
            row[f"{m}_mean_latency_ms"], row[f"{m}_median_latency_ms"] = lat["mean"], lat["median"]
            labelled = [(window_truth(sc, w["phase"]), int(w[f"{m}_binary"])) for w in ws]
            labelled = [(t, p) for t, p in labelled if t is not None]
            row[f"{m}_window_positive_rate"] = _div(sum(p for _, p in labelled), len(labelled))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------- multiclass (ML only)
def multiclass_confusion(windows: Sequence[Dict[str, Any]], classes: Sequence[str]) -> Dict[str, Any]:
    """
    ML window-level confusion over the classes the model actually has. Truth = the scenario, restricted to
    windows that show it (phase EVENT, or NO_EVENT for Normal Driving) and to scenarios the model can
    represent. Scenarios outside `classes` (Rollover, Multi-Impact) are NOT scored; see ood_distribution().
    """
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for w in windows:
        sc = w["scenario"]
        if sc not in idx or w["phase"] not in (S.EVENT, S.NO_EVENT):
            continue
        if w["ml_label"] in idx:
            matrix[idx[sc]][idx[w["ml_label"]]] += 1
    per_class = {}
    for c in classes:
        i = idx[c]
        tp = matrix[i][i]
        support = sum(matrix[i])
        pred = sum(matrix[r][i] for r in range(len(classes)))
        p, r_ = _div(tp, pred), _div(tp, support)
        per_class[c] = {"precision": p, "recall": r_, "support": support,
                        "f1": None if p is None or r_ is None or p + r_ == 0 else 2 * p * r_ / (p + r_)}
    return {"classes": classes, "matrix": matrix, "per_class": per_class}


def ood_distribution(windows: Sequence[Dict[str, Any]], classes: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """What the ML model says for EVENT windows of scenarios it has no class for (reported, never scored)."""
    out: Dict[str, Dict[str, int]] = {}
    for w in windows:
        if w["scenario"] in classes or w["phase"] != S.EVENT:
            continue
        d = out.setdefault(w["scenario"], {})
        d[w["ml_label"]] = d.get(w["ml_label"], 0) + 1
    return out


# ---------------------------------------------------------------------------- paired comparison
def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict[str, Any]:
    """Exact two-sided McNemar test on paired per-recording outcomes (A = first method, B = second)."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired samples differ in length")
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)       # only A right
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)       # only B right
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_first_correct": b, "only_second_correct": c, "discordant": n, "p_value": p,
            "n_pairs": len(a_correct)}


def paired_comparison(recs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ml_ok = [r["ml_binary_decision"] == r["ground_truth"] for r in recs]
    th_ok = [r["threshold_binary_decision"] == r["ground_truth"] for r in recs]
    acc = [r for r in recs if r["ground_truth"] == 1]
    non = [r for r in recs if r["ground_truth"] == 0]
    return {
        "test": "exact McNemar on recording-level correctness (first = ML, second = threshold)",
        "all_recordings": mcnemar_exact(ml_ok, th_ok),
        "accident_recordings_detected": mcnemar_exact([bool(r["ml_binary_decision"]) for r in acc],
                                                      [bool(r["threshold_binary_decision"]) for r in acc]),
        "non_accident_recordings_correct": mcnemar_exact([not r["ml_binary_decision"] for r in non],
                                                         [not r["threshold_binary_decision"] for r in non]),
        "caveat": ("Synthetic simulation, finite number of recordings, paired evaluation. A p-value here describes "
                   "this simulator, not real-world performance, and recordings of one scenario are not independent "
                   "draws from any real population."),
    }


def saturation_effect(recs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Detection of accident recordings split by whether the MPU6050 saturated during the event."""
    out: Dict[str, Any] = {}
    acc = [r for r in recs if r["ground_truth"] == 1]
    for label, group in (("saturated_during_event", [r for r in acc if r["saturated_in_event"] > 0]),
                         ("not_saturated_during_event", [r for r in acc if r["saturated_in_event"] == 0])):
        out[label] = {"n": len(group), **{f"{m}_detection_rate": _div(sum(int(r[f"{m}_binary_decision"]) for r in group), len(group))
                                          for m in S.METHODS}}
    return out
