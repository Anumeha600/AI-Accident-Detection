"""
PHASE 10 -- Replay of saved simulation runs.
------------------------------------------------------------------------------
SIMULATION REPLAY. Recorded synthetic data - not real-world accident data.

A replay REPRODUCES A RECORDED RUN: it emits the saved samples in
chronological order together with the decisions, severity, GPS positions and
emergency-state transitions that were saved at the time. It performs no
prediction and loads no model.

    RECORDED RESULT        everything ReplaySession / build_event_timeline return
    LIVE RECOMPUTATION     recompute_decisions() / compare_recorded_vs_recomputed():
                           an optional, SEPARATE re-run of the current AI,
                           threshold detector and severity_v2 on the saved
                           samples, always labelled as such, never mixed into
                           the recorded result.

decide_window() is the single place that turns one 10-sample window into the
decision block (AI + threshold detector + severity_v2); app.py uses it to
record live runs and recompute_decisions() uses it again to re-check them, so
the logic is not duplicated. It reuses predict_expanded, threshold_detector and
severity_v2 unchanged.
"""

import math
from typing import Any, Dict, List, Optional

import pandas as pd

from recording_store import Recording, RecordingError
from window_features import WINDOW_SIZE

RECORDED_LABEL = "RECORDED RESULT"
RECOMPUTED_LABEL = "LIVE RECOMPUTATION"
REPLAY_BANNER = "SIMULATION REPLAY"
SYNTHETIC_NOTICE = "Recorded synthetic data - not real-world accident data."

INITIAL_STATUS = "MONITORING"
GPS_EVENT_INTERVAL_S = 1.0     # GPS positions change every sample; the timeline lists one per interval

READY, PLAYING, PAUSED, STOPPED, FINISHED = "READY", "PLAYING", "PAUSED", "STOPPED", "FINISHED"


# ---------------------------------------------------------------------------
# decisions for one window (shared by live recording and recomputation)
# ---------------------------------------------------------------------------

def decide_window(window_readings: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    AI classification + threshold-detector result + severity_v2 for one window of
    WINDOW_SIZE readings, in the compact shapes Recorder.append_decision() takes.
    Each block is None if its component is unavailable (e.g. config file missing);
    the AI block is required.
    """
    from predict_expanded import predict_window_event          # Phase 4 model, as used by the dashboard
    from realtime_pipeline import severity_for_label
    import threshold_detector as td
    import severity_v2 as sv

    label, confidence, probs = predict_window_event(window_readings)
    out: Dict[str, Any] = {
        "ai": {"label": label, "confidence": float(confidence), "probabilities": dict(probs)},
        "legacy_severity": severity_for_label(label),
        "threshold": None, "severity": None,
    }
    try:
        out["threshold"] = td.detect_window(window_readings)
    except FileNotFoundError:
        pass
    try:
        out["severity"] = sv.assess_window(window_readings)
    except FileNotFoundError:
        pass
    return out


# ---------------------------------------------------------------------------
# RECORDED event timeline
# ---------------------------------------------------------------------------

def build_event_timeline(rec: Recording) -> List[Dict[str, Any]]:
    """
    Important events of the recorded run, chronological (all labelled RECORDED):
    AI event detected / changed / cleared, threshold event detected / cleared,
    severity level change, GPS update (first sample, then once per
    GPS_EVENT_INTERVAL_S), emergency countdown started / alert cancelled /
    alert sent / other status changes.
    """
    events: List[Dict[str, Any]] = []

    def add(t, idx, kind, text):
        events.append({"t": float(t), "sample_index": idx, "kind": kind, "text": text, "source": RECORDED_LABEL})

    prev_ai: Optional[str] = None
    prev_th = False
    prev_sev: Optional[str] = None
    for d in rec.decisions:
        t, idx = d["t"], d["sample_index"]
        ai = d.get("ai")
        if ai is not None:
            normal = ai["label"] == "Normal Driving"
            if prev_ai is None or normal != (prev_ai == "Normal Driving") or (not normal and ai["label"] != prev_ai):
                if not normal:
                    add(t, idx, "ai_event_detected",
                        f"AI event detected: {ai['label']} (confidence {ai['confidence']:.0%})")
                elif prev_ai is not None:
                    add(t, idx, "ai_event_cleared", "AI: back to Normal Driving")
            prev_ai = ai["label"]
        th = d.get("threshold")
        if th is not None:
            if th["event_detected"] and not prev_th:
                add(t, idx, "threshold_event_detected",
                    "Threshold detector: ACCIDENT (" + ", ".join(th.get("triggered_rules", [])) + ")")
            elif prev_th and not th["event_detected"]:
                add(t, idx, "threshold_event_cleared", "Threshold detector: cleared")
            prev_th = th["event_detected"]
        sv = d.get("severity")
        if sv is not None:
            if prev_sev is not None and sv["severity"] != prev_sev:
                add(t, idx, "severity_change",
                    f"Severity {prev_sev} -> {sv['severity']} (score {sv['score']:.1f})")
            elif prev_sev is None and sv["severity"] != "LOW":
                add(t, idx, "severity_change", f"Severity {sv['severity']} (score {sv['score']:.1f})")
            prev_sev = sv["severity"]

    next_gps_t = -math.inf
    for i, s in enumerate(rec.samples):
        if "gps" in s and s["t"] >= next_gps_t:
            add(s["t"], i, "gps_update", f"GPS {s['gps'][0]:.6f}, {s['gps'][1]:.6f}")
            next_gps_t = s["t"] + GPS_EVENT_INTERVAL_S

    texts = {"ALERT COUNTDOWN": ("emergency_countdown_started", "Emergency countdown started"),
             "ALERT CANCELLED": ("alert_cancelled", "Alert cancelled"),
             "ALERT SENT": ("alert_sent", "Alert sent (SIMULATED - no real message)")}
    for tr in rec.transitions:
        kind, text = texts.get(tr["to"], ("status_change", f"Status {tr['from']} -> {tr['to']}"))
        add(tr["t"], tr["sample_index"], kind, text)

    order = {"emergency_countdown_started": 0}
    events.sort(key=lambda e: (e["t"], e["sample_index"], order.get(e["kind"], 1)))
    return events


# ---------------------------------------------------------------------------
# RECORDED playback
# ---------------------------------------------------------------------------

class ReplaySession:
    """
    Steps through a saved recording. States: READY -> PLAYING <-> PAUSED,
    STOPPED, FINISHED. step() emits recorded frames (RECORDED RESULT) in
    chronological order; nothing is predicted.
    """

    def __init__(self, recording: Recording):
        self.recording = recording
        self._samples = recording.samples
        self._decisions: Dict[int, Dict[str, Any]] = {}
        for d in recording.decisions:                      # last decision wins per sample
            self._decisions[d["sample_index"]] = d
        self._transitions = sorted(recording.transitions, key=lambda tr: (tr["t"], tr["sample_index"]))
        self.timeline = build_event_timeline(recording)
        self.position = 0                                   # number of samples emitted so far
        self.status = READY

    # -- control --------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def play(self) -> None:
        if self.status == FINISHED:
            return
        self.status = PLAYING

    def pause(self) -> None:
        if self.status == PLAYING:
            self.status = PAUSED

    def stop(self) -> None:
        """Halt playback; the position is kept so the final state can still be inspected."""
        if self.status != FINISHED:
            self.status = STOPPED

    def reset(self) -> None:
        self.position = 0
        self.status = READY

    # -- stepping ---------------------------------------------------------------
    def step(self, n: int = 1) -> List[Dict[str, Any]]:
        """Emit up to n next frames (also usable while paused/stopped: manual single-stepping)."""
        frames: List[Dict[str, Any]] = []
        for _ in range(max(0, n)):
            if self.position >= self.n_samples:
                break
            frames.append(self.frame(self.position))
            self.position += 1
        if self.position >= self.n_samples:
            self.status = FINISHED
        return frames

    def run_to_end(self) -> List[Dict[str, Any]]:
        self.reset()
        self.play()
        return self.step(self.n_samples) if self.n_samples else self.step(1)

    @property
    def finished(self) -> bool:
        return self.status == FINISHED

    # -- recorded state ------------------------------------------------------------
    def frame(self, index: int) -> Dict[str, Any]:
        s = self._samples[index]
        return {"source": RECORDED_LABEL, "index": index, "t": s["t"],
                "sample": {k: v for k, v in s.items() if k not in ("t", "gps", "wall_time_utc")},
                "gps": tuple(s["gps"]) if "gps" in s else None,
                "wall_time_utc": s.get("wall_time_utc"),
                "decision": self._decisions.get(index)}

    def state_at(self, index: Optional[int] = None) -> Dict[str, Any]:
        """
        RECORDED state after sample `index` (default: the last emitted sample):
        latest AI / threshold / severity decision, emergency status, GPS, and the
        timeline events up to that point. {} fields are None before any data.
        """
        if index is None:
            index = self.position - 1
        if index < 0 or not self._samples:
            return {"source": RECORDED_LABEL, "index": -1, "elapsed_s": 0.0, "status": INITIAL_STATUS,
                    "ai": None, "threshold": None, "severity": None, "legacy_severity": None, "gps": None,
                    "events": []}
        index = min(index, self.n_samples - 1)
        s = self._samples[index]
        ai = th = sv = legacy = None
        for i in range(index, -1, -1):                       # latest decision at or before index
            d = self._decisions.get(i)
            if d:
                ai, th, sv, legacy = d.get("ai"), d.get("threshold"), d.get("severity"), d.get("legacy_severity")
                break
        status = INITIAL_STATUS
        for tr in self._transitions:
            # A transition belongs to the sample it was recorded at; ones that happened after the
            # stream ended (e.g. the countdown expiring) are attached to the last sample.
            if tr["sample_index"] <= index:
                status = tr["to"]
        gps = next((tuple(self._samples[i]["gps"]) for i in range(index, -1, -1) if "gps" in self._samples[i]), None)
        return {"source": RECORDED_LABEL, "index": index, "elapsed_s": s["t"], "status": status,
                "ai": ai, "threshold": th, "severity": sv, "legacy_severity": legacy, "gps": gps,
                "sample": {k: v for k, v in s.items() if k not in ("t", "gps", "wall_time_utc")},
                "events": [e for e in self.timeline if e["sample_index"] <= index]}

    def sensor_table(self, upto: Optional[int] = None) -> pd.DataFrame:
        """Recorded sensor timeline (t, 8 channels, magnitudes) up to and including sample `upto`."""
        upto = self.position - 1 if upto is None else upto
        rows = self._samples[: max(upto + 1, 0)]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["accel_mag_g"] = (df.accel_x_g ** 2 + df.accel_y_g ** 2 + df.accel_z_g ** 2) ** 0.5
        df["gyro_mag_dps"] = (df.gyro_x_dps ** 2 + df.gyro_y_dps ** 2 + df.gyro_z_dps ** 2) ** 0.5
        return df


# ---------------------------------------------------------------------------
# LIVE RECOMPUTATION (separate and labelled; not part of the recorded result)
# ---------------------------------------------------------------------------

def recompute_decisions(rec: Recording) -> List[Dict[str, Any]]:
    """
    LIVE RECOMPUTATION: run the CURRENT AI model, threshold detector and
    severity_v2 over the recorded samples, window by window (same rolling
    windows the live app used). Returns decisions in the stored shape plus
    "source": "LIVE RECOMPUTATION". The recording itself is never modified.
    """
    out: List[Dict[str, Any]] = []
    cols = [k for k in rec.samples[0] if k not in ("t", "gps", "wall_time_utc")] if rec.samples else []
    for i in range(WINDOW_SIZE - 1, len(rec.samples)):
        window = [{k: rec.samples[j][k] for k in cols} for j in range(i - WINDOW_SIZE + 1, i + 1)]
        d = decide_window(window)
        d.update({"sample_index": i, "t": rec.samples[i]["t"], "source": RECOMPUTED_LABEL})
        out.append(d)
    return out


def compare_recorded_vs_recomputed(rec: Recording, tolerance: float = 1e-9) -> Dict[str, Any]:
    """
    Compare the recorded decisions with a fresh LIVE RECOMPUTATION. Mismatches mean
    the software/model changed since the run was recorded (or the recording was
    edited); they do not invalidate the recorded result.
    """
    recomputed = {d["sample_index"]: d for d in recompute_decisions(rec)}
    mismatches: List[Dict[str, Any]] = []
    compared = 0
    for rd in rec.decisions:
        new = recomputed.get(rd["sample_index"])
        if new is None:
            mismatches.append({"sample_index": rd["sample_index"], "field": "decision", "recorded": "present",
                               "recomputed": "missing"})
            continue
        compared += 1
        checks = [("ai.label", (rd.get("ai") or {}).get("label"), new["ai"]["label"]),
                  ("ai.confidence", (rd.get("ai") or {}).get("confidence"), new["ai"]["confidence"]),
                  ("threshold.event_detected", (rd.get("threshold") or {}).get("event_detected"),
                   (new["threshold"] or {}).get("event_detected")),
                  ("threshold.triggered_rules", (rd.get("threshold") or {}).get("triggered_rules"),
                   (new["threshold"] or {}).get("triggered_rules")),
                  ("severity.severity", (rd.get("severity") or {}).get("severity"),
                   (new["severity"] or {}).get("severity")),
                  ("severity.score", (rd.get("severity") or {}).get("score"), (new["severity"] or {}).get("score"))]
        for name, a, b in checks:
            if a is None:                      # field not recorded: nothing to compare
                continue
            same = (abs(a - b) <= max(tolerance, 5e-4) if isinstance(a, float) and isinstance(b, (int, float))
                    else a == b)
            if not same:
                mismatches.append({"sample_index": rd["sample_index"], "field": name, "recorded": a, "recomputed": b})
    return {"source": RECOMPUTED_LABEL + " vs " + RECORDED_LABEL, "decisions_compared": compared,
            "mismatches": mismatches, "identical": not mismatches}
