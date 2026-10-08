"""
PHASE 10 -- Recording persistence for simulation runs.
------------------------------------------------------------------------------
Every run can be saved as ONE self-contained JSON file (`recordings/<id>.json`)
and loaded back later for replay (see replay.py). No database, no extra
dependency, no absolute filesystem paths inside a recording.

ALL RECORDINGS OF SIMULATED RUNS ARE SYNTHETIC DATA, NOT REAL ACCIDENT DATA.

SCHEMA (schema_version = 1)
    {
      "schema_version": 1,
      "recording_id":  "20261009T101500-severe-accident-3fa81c2e",   # unique, filename-safe
      "created_utc":   "2026-10-09T10:15:00+00:00",
      "finished_utc":  "...",
      "scenario":      "Severe Accident" | "n/a (live hardware)",
      "source_type":   "simulated" | "stm32" | "virtual_hardware" | ...,
      "synthetic":     true,                  # false only for real hardware streams
      "sample_rate_hz": 10.0,
      "duration_s":    5.0,                   # n_samples / sample_rate_hz
      "software": {...}, "models": {...},     # versions / FILE NAMES + md5 (never paths)
      "alert": {"countdown_total_s": 10},     # optional
      "virtual_hardware": {...},              # optional (Phase 10.5): seed, MPU6050 config, notices
      "samples": [ {"t": 0.0, "wall_time_utc": "...", <8 FEATURE_COLUMNS>, "gps": [lat, lon],
                    "hw": {...}} ... ],           # "hw" optional (Phase 10.5): saturation, MCU event, ...
      "decisions": [ {"sample_index": 9, "t": 0.9,
                      "ai": {"label", "confidence", "probabilities"},
                      "threshold": {"event_detected", "classification", "triggered_rules", "event_class"},
                      "severity": {"severity", "score", "triggered_rules"},      # severity_v2
                      "legacy_severity": "HIGH", "status": "ALERT COUNTDOWN"} ... ],
      "state_transitions": [ {"sample_index", "t", "from", "to", "reason"} ... ],
      "summary": {...}                        # derived; recomputable via summarize()
    }
    * Sensor values live ONLY in `samples`; decisions point at them with
      `sample_index` instead of copying values. GPS rides on its sample.
    * `t` is stream time in seconds from the start (sample_index / rate for
      simulated sources, wall-clock for hardware). Transitions that happen
      after the stream ended (e.g. the alert countdown expiring) keep
      advancing from the last sample's `t`.
    * Optional: gps, decisions (and their sub-blocks), state_transitions,
      alert, models, software, summary. Missing optional data loads fine.
    * Newer / unknown schema versions are REJECTED with a clear message.
"""

import hashlib
import json
import math
import os
import platform
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from common import FEATURE_COLUMNS
from severity_v2 import SEVERITY_LEVELS

SCHEMA_VERSION = 1
SOFTWARE_VERSION = "accident-detection-sim phase10"
DEFAULT_RECORDINGS_DIR = "recordings"
RECORDINGS_DIR_ENV = "ACCIDENT_RECORDINGS_DIR"
MAX_FILE_BYTES = 200 * 1024 * 1024
SYNTHETIC_NOTICE = "Recorded synthetic data - not real-world accident data."

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")
_NORMAL_LABEL = "Normal Driving"


class RecordingError(Exception):
    """A recording could not be created, saved, loaded or validated (message is user-presentable)."""


class RecordingNotFound(RecordingError):
    """No recording with that id."""


def default_recordings_dir() -> Path:
    return Path(os.environ.get(RECORDINGS_DIR_ENV, DEFAULT_RECORDINGS_DIR))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_recording_id(scenario: Optional[str], now: Optional[datetime] = None) -> str:
    """`<UTC time>-<scenario slug>-<8 random hex>`: readable, filename-safe, collision-resistant."""
    now = now or _utc_now()
    slug = re.sub(r"[^a-z0-9]+", "-", (scenario or "run").lower()).strip("-")[:40] or "run"
    return f"{now:%Y%m%dT%H%M%S}-{slug}-{uuid.uuid4().hex[:8]}"


def is_valid_recording_id(recording_id: Any) -> bool:
    return isinstance(recording_id, str) and bool(_ID_RE.match(recording_id))


def file_md5(path: Union[str, Path]) -> Optional[str]:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def describe_models(classifier_path: Optional[str] = None, threshold_config: Optional[str] = None,
                    severity_config: Optional[str] = None) -> Dict[str, Any]:
    """Model/config identification by FILE NAME + md5 (never a filesystem path)."""
    out: Dict[str, Any] = {}
    for key, path in (("classifier", classifier_path), ("threshold_config", threshold_config),
                      ("severity_config", severity_config)):
        if path:
            out[key] = {"file": os.path.basename(path), "md5": file_md5(path)}
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_recording_dict(d: Any) -> List[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    if not isinstance(d, dict):
        return ["recording is not a JSON object"]
    errors: List[str] = []
    version = d.get("schema_version")
    if version is None:
        return ["missing required field 'schema_version'"]
    if version != SCHEMA_VERSION:
        return [f"unsupported schema_version {version!r} (this software reads version {SCHEMA_VERSION})"]

    for key in ("recording_id", "created_utc", "scenario", "source_type", "sample_rate_hz", "samples"):
        if key not in d:
            errors.append(f"missing required field '{key}'")
    if errors:
        return errors
    if not is_valid_recording_id(d["recording_id"]):
        errors.append("recording_id is not a safe identifier")
    if not isinstance(d["scenario"], str) or not isinstance(d["source_type"], str) \
            or not isinstance(d["created_utc"], str):
        errors.append("scenario, source_type and created_utc must be strings")
    if not _is_num(d["sample_rate_hz"]) or d["sample_rate_hz"] <= 0:
        errors.append("sample_rate_hz must be a positive number")
    if not isinstance(d["samples"], list):
        return errors + ["'samples' must be a list"]

    prev_t = -math.inf
    for i, s in enumerate(d["samples"]):
        if not isinstance(s, dict):
            errors.append(f"samples[{i}] is not an object")
            break
        bad = [k for k in ["t", *FEATURE_COLUMNS] if not _is_num(s.get(k))]
        if bad:
            errors.append(f"samples[{i}] has missing/non-numeric/non-finite fields: {bad}")
            break
        if s["t"] < prev_t:
            errors.append(f"samples are not chronological at index {i} (t={s['t']} < {prev_t})")
            break
        prev_t = s["t"]
        gps = s.get("gps")
        if gps is not None and not (isinstance(gps, list) and len(gps) == 2 and all(_is_num(v) for v in gps)
                                    and -90 <= gps[0] <= 90 and -180 <= gps[1] <= 180):
            errors.append(f"samples[{i}].gps must be [lat, lon] within valid ranges")
            break
    n = len(d["samples"])

    decisions = d.get("decisions", [])
    if not isinstance(decisions, list):
        errors.append("'decisions' must be a list")
    else:
        for i, dec in enumerate(decisions):
            problem = _decision_problem(dec, n)
            if problem:
                errors.append(f"decisions[{i}]: {problem}")
                break
    transitions = d.get("state_transitions", [])
    if not isinstance(transitions, list):
        errors.append("'state_transitions' must be a list")
    else:
        for i, tr in enumerate(transitions):
            if not (isinstance(tr, dict) and isinstance(tr.get("from"), str) and isinstance(tr.get("to"), str)
                    and _is_num(tr.get("t")) and isinstance(tr.get("sample_index"), int)
                    and not isinstance(tr.get("sample_index"), bool)
                    and (n == 0 or 0 <= tr["sample_index"] < n)):
                errors.append(f"state_transitions[{i}] is malformed (needs from, to, t, valid sample_index)")
                break
    if any(not isinstance(s, dict) or not isinstance(s.get("hw", {}), dict) for s in d["samples"]):
        errors.append("sample 'hw' must be an object")
    for opt, typ in (("models", dict), ("software", dict), ("alert", dict), ("summary", dict),
                     ("virtual_hardware", dict)):
        if opt in d and d[opt] is not None and not isinstance(d[opt], typ):
            errors.append(f"'{opt}' must be an object")
    return errors


def _decision_problem(dec: Any, n_samples: int) -> Optional[str]:
    if not isinstance(dec, dict):
        return "not an object"
    si = dec.get("sample_index")
    if not isinstance(si, int) or isinstance(si, bool) or not 0 <= si < n_samples:
        return "sample_index missing or outside the recorded samples"
    if not _is_num(dec.get("t")):
        return "t missing or non-numeric"
    ai = dec.get("ai")
    if ai is not None:
        if not (isinstance(ai, dict) and isinstance(ai.get("label"), str) and _is_num(ai.get("confidence"))):
            return "ai needs a string label and numeric confidence"
        probs = ai.get("probabilities")
        if probs is not None and not (isinstance(probs, dict) and all(_is_num(v) for v in probs.values())):
            return "ai.probabilities must map class -> number"
    th = dec.get("threshold")
    if th is not None and not (isinstance(th, dict) and isinstance(th.get("event_detected"), bool)
                               and isinstance(th.get("triggered_rules", []), list)):
        return "threshold needs boolean event_detected and a triggered_rules list"
    sv = dec.get("severity")
    if sv is not None and not (isinstance(sv, dict) and sv.get("severity") in SEVERITY_LEVELS
                               and _is_num(sv.get("score"))):
        return f"severity needs a level in {SEVERITY_LEVELS} and a numeric score"
    return None


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarize(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Derived overview of a (valid) recording dict. Pure function of its content."""
    samples, decisions = rec.get("samples", []), rec.get("decisions", [])
    transitions = rec.get("state_transitions", [])

    def mag(s, keys):
        return math.sqrt(sum(s[k] ** 2 for k in keys))

    peak_accel = max((mag(s, ["accel_x_g", "accel_y_g", "accel_z_g"]) for s in samples), default=None)
    peak_gyro = max((mag(s, ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]) for s in samples), default=None)

    ai = [d["ai"] for d in decisions if d.get("ai")]
    counts: Dict[str, int] = {}
    best_conf: Dict[str, float] = {}
    for a in ai:
        counts[a["label"]] = counts.get(a["label"], 0) + 1
        best_conf[a["label"]] = max(best_conf.get(a["label"], 0.0), a["confidence"])
    events = {k: v for k, v in counts.items() if k != _NORMAL_LABEL}
    ai_result = None
    if counts:
        pool = events or counts
        ai_result = max(pool, key=lambda k: (pool[k], best_conf[k]))

    sev = [d["severity"] for d in decisions if d.get("severity")]
    top = max(sev, key=lambda s: (SEVERITY_LEVELS.index(s["severity"]), s["score"]), default=None)
    th = [d["threshold"] for d in decisions if d.get("threshold")]
    rules = sorted({r for t in th for r in t.get("triggered_rules", [])})

    final_status = transitions[-1]["to"] if transitions else next(
        (d["status"] for d in reversed(decisions) if d.get("status")), None)
    outcome = next((t["to"] for t in reversed(transitions)
                    if t["to"] in ("ALERT SENT", "ALERT CANCELLED")), None)
    rate = rec.get("sample_rate_hz") or 1.0
    return {
        "scenario": rec.get("scenario"),
        "source_type": rec.get("source_type"),
        "synthetic": rec.get("synthetic", True),
        "duration_s": len(samples) / rate,
        "n_samples": len(samples),
        "n_decisions": len(decisions),
        "n_transitions": len(transitions),
        "ai_result": ai_result,
        "ai_confidence": best_conf.get(ai_result) if ai_result else None,
        "ai_label_counts": counts,
        "threshold_event_detected": (any(t["event_detected"] for t in th) if th else None),
        "threshold_triggered_rules": rules,
        "severity_max": top["severity"] if top else None,
        "severity_max_score": top["score"] if top else None,
        "peak_accel_g": peak_accel,
        "peak_gyro_dps": peak_gyro,
        "emergency_final_status": final_status,
        "alert_outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Recording object
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    """A validated, immutable-by-convention view of one saved run (`data` is the JSON dict)."""
    data: Dict[str, Any] = field(repr=False)

    @property
    def recording_id(self) -> str:
        return self.data["recording_id"]

    @property
    def scenario(self) -> str:
        return self.data["scenario"]

    @property
    def sample_rate_hz(self) -> float:
        return self.data["sample_rate_hz"]

    @property
    def samples(self) -> List[Dict[str, Any]]:
        return self.data["samples"]

    @property
    def decisions(self) -> List[Dict[str, Any]]:
        return self.data.get("decisions", [])

    @property
    def transitions(self) -> List[Dict[str, Any]]:
        return self.data.get("state_transitions", [])

    @property
    def summary(self) -> Dict[str, Any]:
        return summarize(self.data)

    @classmethod
    def from_dict(cls, d: Any) -> "Recording":
        errors = validate_recording_dict(d)
        if errors:
            raise RecordingError("; ".join(errors[:5]))
        return cls(d)

    def to_dict(self) -> Dict[str, Any]:
        return self.data


class Recorder:
    """
    Accumulates one run in memory; nothing touches disk until finish(save=True).

        rec = store.start_recording("Severe Accident", "simulated", 10.0)
        rec.append_sample(reading, gps=(lat, lon))
        rec.append_decision(ai=..., threshold=..., severity=..., status=...)
        rec.append_transition("MONITORING", "ALERT COUNTDOWN")
        recording = rec.finish()
    """

    def __init__(self, store: "RecordingStore", scenario: Optional[str], source_type: str,
                 sample_rate_hz: float, models: Optional[Dict[str, Any]] = None,
                 alert: Optional[Dict[str, Any]] = None, synthetic: Optional[bool] = None,
                 now: Optional[datetime] = None, recording_id: Optional[str] = None):
        if not _is_num(sample_rate_hz) or sample_rate_hz <= 0:
            raise RecordingError("sample_rate_hz must be a positive number")
        self._store = store
        now = now or _utc_now()
        scenario = scenario if scenario else "n/a (live hardware)"
        simulated = source_type.startswith(("simulated", "virtual"))   # virtual hardware also runs on simulated time
        is_synthetic = simulated if synthetic is None else bool(synthetic)
        self._data: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "recording_id": recording_id or store.unique_id(scenario, now),   # explicit id: Phase 11 experiments
            "created_utc": now.isoformat(),
            "scenario": scenario,
            "source_type": source_type,
            "synthetic": is_synthetic,
            "notice": SYNTHETIC_NOTICE if is_synthetic
            else "Live hardware stream (accelerometer/gyro real; vibration, speed, GPS and alert are not).",
            "sample_rate_hz": float(sample_rate_hz),
            "software": {"version": SOFTWARE_VERSION, "python": platform.python_version()},
            "models": models or {},
            "samples": [], "decisions": [], "state_transitions": [],
        }
        if alert:
            self._data["alert"] = dict(alert)
        self._finished = False
        self._wall_start = time.monotonic()
        self._last_sample_wall = self._wall_start
        self._wall_clock_time = not simulated

    # -- appending ----------------------------------------------------------
    def _check_open(self) -> None:
        if self._finished:
            raise RecordingError("recording is already finished")

    @property
    def recording_id(self) -> str:
        return self._data["recording_id"]

    @property
    def n_samples(self) -> int:
        return len(self._data["samples"])

    @property
    def last_t(self) -> float:
        s = self._data["samples"]
        return s[-1]["t"] if s else 0.0

    def append_sample(self, reading: Dict[str, float], gps: Optional[tuple] = None,
                      t: Optional[float] = None, hardware: Optional[Dict[str, Any]] = None) -> int:
        """Append one reading (all FEATURE_COLUMNS). Returns its sample_index. `hardware` is optional per-sample info."""
        self._check_open()
        missing = [k for k in FEATURE_COLUMNS if not _is_num(reading.get(k))]
        if missing:
            raise RecordingError(f"sample is missing/non-numeric/non-finite fields: {missing}")
        idx = self.n_samples
        now = time.monotonic()
        if t is None:
            t = (now - self._wall_start) if self._wall_clock_time else idx / self._data["sample_rate_hz"]
        if not _is_num(t) or (idx and t < self.last_t):
            raise RecordingError("sample time must be finite and non-decreasing")
        sample: Dict[str, Any] = {"t": float(t), "wall_time_utc": _utc_now().isoformat()}
        sample.update({k: float(reading[k]) for k in FEATURE_COLUMNS})
        if gps is not None:
            lat, lon = float(gps[0]), float(gps[1])
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise RecordingError("gps must be a finite (lat, lon) within valid ranges")
            sample["gps"] = [lat, lon]
        if hardware:
            sample["hw"] = dict(hardware)
        self._data["samples"].append(sample)
        self._last_sample_wall = now
        return idx

    def append_decision(self, ai: Optional[Dict[str, Any]] = None, threshold: Optional[Dict[str, Any]] = None,
                        severity: Optional[Dict[str, Any]] = None, legacy_severity: Optional[str] = None,
                        status: Optional[str] = None, sample_index: Optional[int] = None) -> None:
        """Attach the system's decisions for the latest (or given) sample. Only compact fields are kept."""
        self._check_open()
        if not self._data["samples"]:
            raise RecordingError("cannot append a decision before any sample")
        idx = self.n_samples - 1 if sample_index is None else sample_index
        if not 0 <= idx < self.n_samples:
            raise RecordingError("sample_index outside the recorded samples")
        dec: Dict[str, Any] = {"sample_index": idx, "t": self._data["samples"][idx]["t"]}
        if ai is not None:
            dec["ai"] = {"label": str(ai["label"]), "confidence": float(ai["confidence"])}
            if ai.get("probabilities") is not None:
                dec["ai"]["probabilities"] = {str(k): float(v) for k, v in ai["probabilities"].items()}
        if threshold is not None:
            dec["threshold"] = {"event_detected": bool(threshold["event_detected"]),
                                "classification": threshold.get("classification"),
                                "triggered_rules": list(threshold.get("triggered_rules", [])),
                                "event_class": threshold.get("event_class")}
        if severity is not None:
            dec["severity"] = {"severity": severity["severity"], "score": float(severity["score"]),
                               "triggered_rules": list(severity.get("triggered_rules", []))}
        if legacy_severity is not None:
            dec["legacy_severity"] = legacy_severity
        if status is not None:
            dec["status"] = status
        problem = _decision_problem(dec, self.n_samples)
        if problem:
            raise RecordingError(f"invalid decision: {problem}")
        self._data["decisions"].append(dec)

    def append_transition(self, from_state: str, to_state: str, reason: str = "", t: Optional[float] = None) -> None:
        """
        Record an emergency/status state change. `t` defaults to the last
        sample's t plus the wall-clock time since that sample, so transitions
        after the stream ended (countdown expiry) keep advancing.
        """
        self._check_open()
        if t is None:
            t = self.last_t + (time.monotonic() - self._last_sample_wall)
        self._data["state_transitions"].append({
            "sample_index": max(self.n_samples - 1, 0), "t": float(t),
            "from": from_state, "to": to_state, "reason": reason,
        })

    def set_virtual_hardware(self, info: Dict[str, Any]) -> None:
        """Run-level virtual-hardware metadata (seed, sensor configuration, honesty notices)."""
        self._check_open()
        self._data["virtual_hardware"] = dict(info)

    def set_alert_config(self, countdown_total_s: int) -> None:
        self._check_open()
        self._data["alert"] = {"countdown_total_s": int(countdown_total_s)}

    # -- finishing ------------------------------------------------------------
    def finish(self, save: bool = True) -> Recording:
        """Seal the recording (computes duration + summary) and, by default, save it."""
        self._check_open()
        self._finished = True
        d = self._data
        d["finished_utc"] = _utc_now().isoformat()
        d["duration_s"] = len(d["samples"]) / d["sample_rate_hz"]
        d["summary"] = summarize(d)
        recording = Recording.from_dict(d)
        if save:
            self._store.save(recording)
        return recording

    def discard(self) -> None:
        self._finished = True


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class RecordingStore:
    """Directory of `<recording_id>.json` files. Safe against bad ids, corrupt files and overwrites."""

    def __init__(self, directory: Union[str, Path, None] = None):
        self.directory = Path(directory) if directory is not None else default_recordings_dir()

    def _path(self, recording_id: str) -> Path:
        if not is_valid_recording_id(recording_id):
            raise RecordingError(f"invalid recording id {recording_id!r}")
        return self.directory / f"{recording_id}.json"

    def unique_id(self, scenario: Optional[str], now: Optional[datetime] = None) -> str:
        for _ in range(100):
            rid = new_recording_id(scenario, now)
            if not self._path(rid).exists():
                return rid
        raise RecordingError("could not generate a unique recording id")

    def start_recording(self, scenario: Optional[str], source_type: str, sample_rate_hz: float,
                        models: Optional[Dict[str, Any]] = None, alert: Optional[Dict[str, Any]] = None,
                        synthetic: Optional[bool] = None, recording_id: Optional[str] = None) -> Recorder:
        if recording_id is not None and not is_valid_recording_id(recording_id):
            raise RecordingError(f"invalid recording id {recording_id!r}")
        return Recorder(self, scenario, source_type, sample_rate_hz, models, alert, synthetic,
                        recording_id=recording_id)

    def save(self, recording: Recording, overwrite: bool = False) -> Path:
        """Atomic write (temp file + rename). Refuses to replace an existing recording unless asked."""
        data = recording.to_dict()
        errors = validate_recording_dict(data)
        if errors:
            raise RecordingError("refusing to save an invalid recording: " + "; ".join(errors[:3]))
        path = self._path(data["recording_id"])
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if path.exists() and not overwrite:
                raise RecordingError(f"recording {data['recording_id']} already exists")
            text = json.dumps(data, indent=1, allow_nan=False)
            tmp = path.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except (OSError, ValueError, TypeError) as exc:
            raise RecordingError(f"could not save recording: {exc}") from exc
        return path

    def load(self, recording_id: str) -> Recording:
        path = self._path(recording_id)
        return self._load_path(path)

    @staticmethod
    def _load_path(path: Path) -> Recording:
        if not path.is_file():
            raise RecordingNotFound(f"recording file not found: {path.name}")
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise RecordingError(f"{path.name}: file too large")
            raw = json.loads(path.read_text(encoding="utf-8"))
        except RecordingError:
            raise
        except (OSError, UnicodeDecodeError) as exc:
            raise RecordingError(f"{path.name}: cannot be read ({exc})") from exc
        except ValueError as exc:                       # json.JSONDecodeError
            raise RecordingError(f"{path.name}: not valid JSON ({exc})") from exc
        try:
            rec = Recording.from_dict(raw)
        except RecordingError as exc:
            raise RecordingError(f"{path.name}: {exc}") from exc
        if rec.recording_id != path.stem:
            raise RecordingError(f"{path.name}: recording_id inside the file does not match its file name")
        return rec

    def list_recordings(self) -> List[Dict[str, Any]]:
        """
        Newest first. Corrupt files do NOT raise: they appear as
        {"recording_id": <file stem>, "error": <why>} so the UI can show them.
        """
        if not self.directory.is_dir():
            return []
        rows: List[Dict[str, Any]] = []
        for path in self.directory.glob("*.json"):
            try:
                rec = self._load_path(path)
            except RecordingError as exc:
                rows.append({"recording_id": path.stem, "file": path.name, "error": str(exc)})
                continue
            s = rec.summary
            rows.append({"recording_id": rec.recording_id, "file": path.name, "error": None,
                         "created_utc": rec.data["created_utc"], "scenario": rec.scenario,
                         "source_type": rec.data["source_type"], "sample_rate_hz": rec.sample_rate_hz,
                         "duration_s": s["duration_s"], "n_samples": s["n_samples"], "ai_result": s["ai_result"],
                         "severity_max": s["severity_max"], "alert_outcome": s["alert_outcome"],
                         "synthetic": s["synthetic"]})
        rows.sort(key=lambda r: (r.get("created_utc") or "", r["recording_id"]), reverse=True)
        return rows

    def delete(self, recording_id: str) -> bool:
        """
        Delete ONE recording by id. Safe: the id must match the strict pattern,
        the target must be a regular (non-symlink) `.json` file directly inside
        the store directory. Returns False if there was nothing to delete.
        """
        path = self._path(recording_id)
        if not path.exists() and not path.is_symlink():
            return False
        if path.is_symlink() or not path.is_file() or path.resolve().parent != self.directory.resolve():
            raise RecordingError("refusing to delete: not a regular recording file inside the store")
        path.unlink()
        return True
