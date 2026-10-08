"""Phase 10: recording_store (persistence, validation, isolation) and replay (RECORDED playback)."""

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

import recording_store as rs
import replay as rp
from common import FEATURE_COLUMNS
from realtime_pipeline import GPS_ORIGIN, RollingWindowBuffer, next_status, severity_for_label, simulate_gps_step
from recording_store import Recording, RecordingError, RecordingNotFound, RecordingStore
from sensor_source import SimulatedSensorSource

ROOT = Path(__file__).resolve().parent.parent


def reading(i=0, **over):
    r = {c: 0.0 for c in FEATURE_COLUMNS}
    r.update(accel_z_g=1.0, speed_kmh=50.0 - i, vibration_level=0.3)
    r.update(over)
    return r


def fake_decision(label="Normal Driving", conf=0.9, sev="LOW", score=1.0, th=False, rules=()):
    return dict(
        ai={"label": label, "confidence": conf, "probabilities": {label: conf, "Other": 1 - conf}},
        threshold={"event_detected": th, "classification": "ACCIDENT" if th else "NORMAL",
                   "triggered_rules": list(rules), "event_class": label},
        severity={"severity": sev, "score": score, "triggered_rules": []},
        legacy_severity="NORMAL",
    )


def synthetic_recording(store, n=12, with_alert=True, save=True):
    """A small hand-built recording that exercises every field (no ML involved)."""
    rec = store.start_recording("Severe Accident", "simulated", 10.0, models={"classifier": {"file": "m.joblib"}},
                                alert={"countdown_total_s": 10})
    for i in range(n):
        rec.append_sample(reading(i), gps=(GPS_ORIGIN[0] + i * 1e-4, GPS_ORIGIN[1]))
        if i >= 9:
            hot = i >= 10
            rec.append_decision(**fake_decision("Severe Accident" if hot else "Normal Driving",
                                                sev="CRITICAL" if hot else "LOW", score=90.0 if hot else 2.0,
                                                th=hot, rules=["impact_acceleration"] if hot else []),
                                status="ALERT COUNTDOWN" if hot else "MONITORING")
        if i == 10 and with_alert:
            rec.append_transition("MONITORING", "ALERT COUNTDOWN", reason="HIGH severity")
    if with_alert:
        rec.append_transition("ALERT COUNTDOWN", "ALERT SENT", reason="countdown expired",
                              t=rec.last_t + 10.0)
    return rec.finish(save=save)


@pytest.fixture
def store(tmp_path):
    return RecordingStore(tmp_path / "recs")


# =============================== creation / saving / loading ==================================

def test_create_append_finish_basic_fields(store):
    r = synthetic_recording(store, save=False)
    d = r.to_dict()
    assert d["schema_version"] == 1 and d["source_type"] == "simulated" and d["scenario"] == "Severe Accident"
    assert d["sample_rate_hz"] == 10.0 and d["synthetic"] is True
    assert d["duration_s"] == pytest.approx(1.2) and len(d["samples"]) == 12
    assert "synthetic" in d["notice"].lower() and "not real-world" in d["notice"].lower()
    assert [s["t"] for s in d["samples"]] == pytest.approx([i / 10 for i in range(12)])
    assert all(set(FEATURE_COLUMNS) <= set(s) and "gps" in s for s in d["samples"])
    assert d["decisions"][0]["sample_index"] == 9 and len(d["decisions"]) == 3
    assert d["state_transitions"][0]["to"] == "ALERT COUNTDOWN"
    assert not (store.directory / f"{d['recording_id']}.json").exists()          # not saved yet


def test_decisions_reference_samples_instead_of_copying_sensor_values(store):
    d = synthetic_recording(store).to_dict()
    for dec in d["decisions"]:
        assert not (set(FEATURE_COLUMNS) & set(dec))
        assert 0 <= dec["sample_index"] < len(d["samples"])
        assert dec["t"] == d["samples"][dec["sample_index"]]["t"]


def test_save_and_load_round_trip_is_exact(store):
    saved = synthetic_recording(store)
    loaded = store.load(saved.recording_id)
    assert loaded.to_dict() == saved.to_dict()
    assert loaded.summary == saved.summary
    assert json.loads(json.dumps(saved.to_dict())) == saved.to_dict()        # float round trip is lossless


def test_round_trip_of_a_real_pipeline_run(store):
    rec = run_pipeline_recording(store, seed=3, scenario="Severe Accident")
    again = store.load(rec.recording_id)
    assert again.to_dict() == rec.to_dict()
    assert len(again.samples) == 50 and len(again.decisions) == 41


def test_saved_file_has_no_absolute_paths(store):
    r = synthetic_recording(store)
    text = (store.directory / f"{r.recording_id}.json").read_text()
    assert str(store.directory) not in text and ":\\" not in text and "/tmp/" not in text
    assert os.path.basename(__file__) not in text


def test_models_metadata_uses_file_names_not_paths():
    m = rs.describe_models(str(ROOT / "accident_classifier_model_expanded.joblib"),
                           str(ROOT / "threshold_config_v1.json"), str(ROOT / "severity_config_v2.json"))
    assert m["classifier"]["file"] == "accident_classifier_model_expanded.joblib"
    assert len(m["classifier"]["md5"]) == 32 and set(m) == {"classifier", "threshold_config", "severity_config"}
    assert not any("\\" in v["file"] or "/" in v["file"] for v in m.values())


def test_save_does_not_overwrite_and_leaves_no_temp_files(store):
    r = synthetic_recording(store)
    with pytest.raises(RecordingError, match="already exists"):
        store.save(r)
    store.save(r, overwrite=True)
    assert [p.name for p in store.directory.iterdir()] == [f"{r.recording_id}.json"]


def test_cannot_save_invalid_recording(store):
    bad = synthetic_recording(store, save=False).to_dict()
    bad["samples"][3]["accel_x_g"] = float("nan")
    with pytest.raises(RecordingError):
        store.save(Recording(bad))


def test_zero_length_recording_round_trips(store):
    rec = store.start_recording("Normal Driving", "simulated", 10.0).finish()
    loaded = store.load(rec.recording_id)
    assert loaded.samples == [] and loaded.summary["duration_s"] == 0 and loaded.summary["ai_result"] is None
    assert loaded.summary["peak_accel_g"] is None


# =============================== ids / isolation ===============================================

def test_recording_ids_are_unique_safe_and_not_just_the_scenario():
    ids = {rs.new_recording_id("Severe Accident") for _ in range(2000)}
    assert len(ids) == 2000
    sample = next(iter(ids))
    assert sample != "Severe Accident" and "severe-accident" in sample and rs.is_valid_recording_id(sample)


def test_same_second_same_scenario_recordings_do_not_collide(store):
    ids = {synthetic_recording(store, n=2, with_alert=False).recording_id for _ in range(25)}
    assert len(ids) == 25 and len(list(store.directory.glob("*.json"))) == 25


@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "", ".hidden", "x" * 200, "has space", "ok.json", None, 5])
def test_unsafe_ids_rejected(store, bad):
    assert not rs.is_valid_recording_id(bad)
    with pytest.raises(RecordingError):
        store.load(bad)
    with pytest.raises(RecordingError):
        store.delete(bad)


def test_multiple_recordings_are_isolated(store):
    a = store.start_recording("Pothole", "simulated", 10.0)
    b = store.start_recording("Rollover", "simulated", 10.0)
    for i in range(5):                              # interleaved appends
        a.append_sample(reading(i, speed_kmh=10.0))
        b.append_sample(reading(i, speed_kmh=99.0))
    ra, rb = a.finish(), b.finish()
    la, lb = store.load(ra.recording_id), store.load(rb.recording_id)
    assert {s["speed_kmh"] for s in la.samples} == {10.0} and {s["speed_kmh"] for s in lb.samples} == {99.0}
    assert la.scenario == "Pothole" and lb.scenario == "Rollover" and la.recording_id != lb.recording_id


def test_list_recordings_newest_first_and_metadata(store):
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = []
    for k in range(3):
        rec = rs.Recorder(store, "Hard Braking", "simulated", 10.0, now=base + timedelta(minutes=k))
        rec.append_sample(reading())
        ids.append(rec.finish().recording_id)
    rows = store.list_recordings()
    assert [r["recording_id"] for r in rows] == ids[::-1]
    assert rows[0]["scenario"] == "Hard Braking" and rows[0]["n_samples"] == 1 and rows[0]["error"] is None


def test_list_on_missing_directory_is_empty(tmp_path):
    assert RecordingStore(tmp_path / "nope").list_recordings() == []


def test_delete_removes_only_that_recording(store):
    a, b = synthetic_recording(store), synthetic_recording(store)
    assert store.delete(a.recording_id) is True
    assert store.delete(a.recording_id) is False
    assert [r["recording_id"] for r in store.list_recordings()] == [b.recording_id]
    with pytest.raises(RecordingNotFound):
        store.load(a.recording_id)


def test_delete_never_touches_non_recording_files(store):
    synthetic_recording(store)
    other = store.directory / "notes.txt"
    other.write_text("keep me")
    with pytest.raises(RecordingError):
        store.delete("../notes")
    assert other.exists()


# =============================== recorder misuse ================================================

def test_recorder_rejects_misuse(store):
    rec = store.start_recording("X", "simulated", 10.0)
    with pytest.raises(RecordingError):
        rec.append_decision(**fake_decision())                       # before any sample
    with pytest.raises(RecordingError):
        rec.append_sample({"accel_x_g": 1.0})                        # missing channels
    with pytest.raises(RecordingError):
        rec.append_sample(reading(accel_x_g=float("nan")))
    with pytest.raises(RecordingError):
        rec.append_sample(reading(), gps=(95.0, 0.0))
    rec.append_sample(reading())
    with pytest.raises(RecordingError):
        rec.append_sample(reading(), t=-1.0)                         # time going backwards
    with pytest.raises(RecordingError):
        rec.append_decision(**fake_decision(), sample_index=5)
    with pytest.raises(RecordingError):
        rec.append_decision(ai={"label": "x", "confidence": 1.0}, severity={"severity": "EXTREME", "score": 1})
    rec.finish(save=False)
    with pytest.raises(RecordingError, match="finished"):
        rec.append_sample(reading())
    with pytest.raises(RecordingError):
        rs.Recorder(store, "X", "simulated", 0)


def test_optional_blocks_may_be_omitted_when_recording(store):
    rec = store.start_recording(None, "stm32", 10.0)
    rec.append_sample(reading())
    rec.append_decision(ai={"label": "Normal Driving", "confidence": 0.8})        # no probs/threshold/severity
    r = rec.finish()
    d = store.load(r.recording_id).to_dict()
    assert d["scenario"].startswith("n/a") and d["synthetic"] is False and "hardware" in d["notice"].lower()
    assert "probabilities" not in d["decisions"][0]["ai"] and "threshold" not in d["decisions"][0]


# =============================== validation / corrupt files ==================================

@pytest.fixture
def good(store):
    return synthetic_recording(store).to_dict()


def errors_of(d):
    return rs.validate_recording_dict(d)


def test_valid_recording_has_no_errors(good):
    assert errors_of(good) == []


def test_schema_version_checks(good):
    for v in (0, 2, 99, "1", None):
        bad = {**good, "schema_version": v}
        assert errors_of(bad) and "schema_version" in errors_of(bad)[0]
    no_version = {k: v for k, v in good.items() if k != "schema_version"}
    assert "schema_version" in errors_of(no_version)[0]
    assert "unsupported" in errors_of({**good, "schema_version": 2})[0]


@pytest.mark.parametrize("field", ["recording_id", "created_utc", "scenario", "source_type", "sample_rate_hz", "samples"])
def test_missing_required_field_reported(good, field):
    bad = {k: v for k, v in good.items() if k != field}
    assert any(field in e for e in errors_of(bad))


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.update(sample_rate_hz=0), "sample_rate_hz"),
    (lambda d: d.update(sample_rate_hz="fast"), "sample_rate_hz"),
    (lambda d: d.update(recording_id="../evil"), "recording_id"),
    (lambda d: d.update(samples="nope"), "samples"),
    (lambda d: d["samples"][2].pop("speed_kmh"), "samples[2]"),
    (lambda d: d["samples"][2].update(accel_x_g=float("inf")), "samples[2]"),
    (lambda d: d["samples"][2].update(t=-5.0), "chronological"),
    (lambda d: d["samples"][1].update(gps=[999, 0]), "gps"),
    (lambda d: d["samples"].__setitem__(0, 7), "samples[0]"),
    (lambda d: d["decisions"][0].update(sample_index=999), "decisions[0]"),
    (lambda d: d["decisions"][0].update(severity={"severity": "EXTREME", "score": 1}), "decisions[0]"),
    (lambda d: d["decisions"][0].update(ai={"label": 5, "confidence": 1}), "decisions[0]"),
    (lambda d: d["decisions"][0].update(threshold={"event_detected": "yes"}), "decisions[0]"),
    (lambda d: d.update(decisions={}), "decisions"),
    (lambda d: d["state_transitions"][0].pop("to"), "state_transitions[0]"),
    (lambda d: d["state_transitions"][0].update(sample_index=500), "state_transitions[0]"),
    (lambda d: d.update(models=[1]), "models"),
])
def test_malformed_content_is_rejected_with_a_useful_message(good, mutate, expect):
    bad = copy.deepcopy(good)
    mutate(bad)
    errs = errors_of(bad)
    assert errs and any(expect in e for e in errs), errs
    with pytest.raises(RecordingError, match=".+"):
        Recording.from_dict(bad)


@pytest.mark.parametrize("junk", [None, [], "text", 5, []])
def test_non_object_top_level_rejected(junk):
    assert errors_of(junk) == ["recording is not a JSON object"]


def test_missing_optional_fields_still_load(good, store):
    minimal = {k: good[k] for k in ("schema_version", "recording_id", "created_utc", "scenario", "source_type",
                                     "sample_rate_hz", "samples")}
    for s in minimal["samples"]:
        s.pop("gps", None)
        s.pop("wall_time_utc", None)
    rec = Recording.from_dict(minimal)
    assert rec.decisions == [] and rec.transitions == []
    s = rec.summary
    assert s["n_samples"] == 12 and s["ai_result"] is None and s["severity_max"] is None
    path = store.directory / f"{minimal['recording_id']}.json"
    store.directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(minimal))
    session = rp.ReplaySession(store.load(minimal["recording_id"]))
    assert len(session.run_to_end()) == 12 and session.state_at()["gps"] is None


def write_file(store, name, text):
    store.directory.mkdir(parents=True, exist_ok=True)
    (store.directory / name).write_bytes(text if isinstance(text, bytes) else text.encode())


@pytest.mark.parametrize("name,content", [
    ("trunc.json", '{"schema_version": 1, "recording_id": "trunc", "samples": ['),
    ("empty.json", ""),
    ("binary.json", b"\xff\xfe\x00\x01garbage"),
    ("list.json", "[1,2,3]"),
    ("nan.json", '{"schema_version": 1, "x": NaN}'),
    ("wrongver.json", json.dumps({"schema_version": 7, "recording_id": "wrongver"})),
    ("nofields.json", json.dumps({"schema_version": 1, "recording_id": "nofields"})),
])
def test_corrupted_files_fail_gracefully(store, name, content):
    write_file(store, name, content)
    stem = name[:-5]
    with pytest.raises(RecordingError) as exc:
        store.load(stem)
    assert stem in str(exc.value) or "json" in str(exc.value).lower() or "schema" in str(exc.value).lower()
    rows = store.list_recordings()                       # listing must not raise
    assert rows[0]["error"] and rows[0]["recording_id"] == stem


def test_one_corrupt_file_does_not_hide_good_recordings(store):
    good_rec = synthetic_recording(store)
    write_file(store, "broken.json", "{not json")
    rows = store.list_recordings()
    assert {r["recording_id"] for r in rows} == {good_rec.recording_id, "broken"}
    assert [r for r in rows if r["error"]][0]["recording_id"] == "broken"
    assert store.load(good_rec.recording_id).recording_id == good_rec.recording_id


def test_id_inside_file_must_match_file_name(store):
    r = synthetic_recording(store)
    src = store.directory / f"{r.recording_id}.json"
    renamed = store.directory / "other-name.json"
    renamed.write_text(src.read_text())
    with pytest.raises(RecordingError, match="does not match"):
        store.load("other-name")


def test_missing_recording_is_reported_as_not_found(store):
    with pytest.raises(RecordingNotFound):
        store.load("20260101T000000-nothing-deadbeef")


# =============================== summary ==========================================================

def test_summary_values(store):
    s = synthetic_recording(store).summary
    assert s["scenario"] == "Severe Accident" and s["synthetic"] is True
    assert s["duration_s"] == pytest.approx(1.2) and s["n_samples"] == 12
    assert s["ai_result"] == "Severe Accident" and s["ai_confidence"] == 0.9     # event labels outrank Normal
    assert s["ai_label_counts"] == {"Normal Driving": 1, "Severe Accident": 2}
    assert s["threshold_event_detected"] is True and s["threshold_triggered_rules"] == ["impact_acceleration"]
    assert s["severity_max"] == "CRITICAL" and s["severity_max_score"] == 90.0
    assert s["peak_accel_g"] == pytest.approx(1.0) and s["peak_gyro_dps"] == 0.0
    assert s["emergency_final_status"] == "ALERT SENT" and s["alert_outcome"] == "ALERT SENT"


def test_summary_alert_cancelled(store):
    rec = store.start_recording("Severe Accident", "simulated", 10.0)
    rec.append_sample(reading())
    rec.append_transition("MONITORING", "ALERT COUNTDOWN")
    rec.append_transition("ALERT COUNTDOWN", "ALERT CANCELLED")
    assert rec.finish().summary["alert_outcome"] == "ALERT CANCELLED"


# =============================== replay ============================================================

def test_replay_is_chronological_and_reproduces_recorded_values(store):
    rec = synthetic_recording(store)
    frames = rp.ReplaySession(store.load(rec.recording_id)).run_to_end()
    assert [f["index"] for f in frames] == list(range(12))
    ts = [f["t"] for f in frames]
    assert ts == sorted(ts) and ts == [s["t"] for s in rec.samples]
    for f, s in zip(frames, rec.samples):
        assert f["source"] == rp.RECORDED_LABEL
        assert f["sample"] == {k: s[k] for k in FEATURE_COLUMNS}
        assert f["gps"] == tuple(s["gps"])
    assert [f["decision"]["sample_index"] for f in frames if f["decision"]] == [9, 10, 11]
    assert frames[10]["decision"] == rec.decisions[1]            # exactly the recorded decision


def test_replay_state_reproduces_ai_threshold_severity_gps_and_emergency_states(store):
    rec = synthetic_recording(store)
    sess = rp.ReplaySession(rec)
    assert sess.state_at(0)["status"] == "MONITORING" and sess.state_at(0)["ai"] is None
    s9 = sess.state_at(9)
    assert s9["ai"]["label"] == "Normal Driving" and s9["severity"]["severity"] == "LOW"
    s10 = sess.state_at(10)
    assert s10["ai"]["label"] == "Severe Accident" and s10["threshold"]["event_detected"] is True
    assert s10["severity"]["severity"] == "CRITICAL" and s10["status"] == "ALERT COUNTDOWN"
    assert s10["gps"] == tuple(rec.samples[10]["gps"])
    last = sess.state_at(11)
    assert last["status"] == "ALERT SENT"                        # post-stream transition shown at the end
    assert sess.state_at(10)["status"] != "ALERT SENT"


def test_event_timeline_contains_all_event_kinds_in_order(store):
    kinds = [e["kind"] for e in rp.build_event_timeline(synthetic_recording(store))]
    for k in ("ai_event_detected", "threshold_event_detected", "severity_change", "gps_update",
              "emergency_countdown_started", "alert_sent"):
        assert k in kinds, k
    ts = [e["t"] for e in rp.build_event_timeline(synthetic_recording(store))]
    assert ts == sorted(ts)
    assert all(e["source"] == rp.RECORDED_LABEL for e in rp.build_event_timeline(synthetic_recording(store)))


def test_timeline_marks_cancelled_alert(store):
    rec = store.start_recording("Severe Accident", "simulated", 10.0)
    rec.append_sample(reading())
    rec.append_transition("MONITORING", "ALERT COUNTDOWN")
    rec.append_transition("ALERT COUNTDOWN", "ALERT CANCELLED")
    kinds = [e["kind"] for e in rp.build_event_timeline(rec.finish())]
    assert kinds.index("emergency_countdown_started") < kinds.index("alert_cancelled")


def test_gps_events_are_throttled(store):
    ev = [e for e in rp.build_event_timeline(synthetic_recording(store, n=50, with_alert=False)) if e["kind"] == "gps_update"]
    assert 4 <= len(ev) <= 6          # 5 s of data, one per second (plus the first)


def test_playback_state_machine(store):
    s = rp.ReplaySession(synthetic_recording(store))
    assert s.status == rp.READY and s.position == 0
    s.play()
    assert s.status == rp.PLAYING
    assert len(s.step(3)) == 3 and s.position == 3
    s.pause()
    assert s.status == rp.PAUSED
    assert len(s.step(1)) == 1 and s.position == 4 and s.status == rp.PAUSED      # manual single step
    s.play()
    s.stop()
    assert s.status == rp.STOPPED and s.position == 4
    s.reset()
    assert s.status == rp.READY and s.position == 0 and s.state_at()["status"] == "MONITORING"
    assert s.step(100)[-1]["index"] == 11 and s.finished and s.status == rp.FINISHED
    assert s.step(5) == []                                                       # nothing after completion
    s.play()
    assert s.status == rp.FINISHED
    s.reset()
    assert [f["index"] for f in s.step(2)] == [0, 1]                             # replays identically after reset


def test_replay_twice_gives_identical_frames(store):
    s = rp.ReplaySession(synthetic_recording(store))
    assert s.run_to_end() == s.run_to_end()


def test_pause_only_applies_while_playing(store):
    s = rp.ReplaySession(synthetic_recording(store))
    s.pause()
    assert s.status == rp.READY


def test_replay_of_empty_recording(store):
    s = rp.ReplaySession(store.start_recording("X", "simulated", 10.0).finish())
    assert s.n_samples == 0 and s.timeline == []
    s.play()
    assert s.step(3) == [] and s.finished
    assert s.state_at()["status"] == "MONITORING" and s.sensor_table().empty
    assert s.run_to_end() == []


def test_replay_loads_no_model_and_predicts_nothing(store, monkeypatch):
    rec = synthetic_recording(store)
    monkeypatch.setattr(rp, "decide_window", lambda *a, **k: (_ for _ in ()).throw(AssertionError("predicted!")))
    s = rp.ReplaySession(rec)
    s.run_to_end()
    s.state_at(5)
    assert s.finished


def test_replay_does_not_mutate_the_recording(store):
    rec = synthetic_recording(store)
    before = copy.deepcopy(rec.to_dict())
    s = rp.ReplaySession(rec)
    s.run_to_end()
    s.state_at(); s.sensor_table(); s.reset()
    assert rec.to_dict() == before


def test_sensor_table_has_magnitudes(store):
    s = rp.ReplaySession(synthetic_recording(store))
    s.step(4)
    t = s.sensor_table()
    assert len(t) == 4 and t.accel_mag_g.iloc[0] == pytest.approx(1.0)


# --------------------- real pipeline run, recording and LIVE RECOMPUTATION --------------------------

def run_pipeline_recording(store, seed=1, scenario="Severe Accident"):
    """Do what app.py does for one run, with a simulated clock instead of Streamlit."""
    src = SimulatedSensorSource(seed=seed)
    src.start(scenario)
    buf, rng = RollingWindowBuffer(), np.random.default_rng(seed)
    lat, lon = GPS_ORIGIN
    rec = store.start_recording(scenario, "simulated", 10.0, alert={"countdown_total_s": 10})
    status = "MONITORING"
    while (r := src.read()) is not None:
        buf.push(r)
        lat, lon = simulate_gps_step(rng, lat, lon)
        rec.append_sample(r, gps=(lat, lon))
        if buf.is_full():
            d = rp.decide_window(buf.as_list())
            new = next_status(status, severity_for_label(d["ai"]["label"]))
            if new != status:
                rec.append_transition(status, new)
            status = new
            rec.append_decision(ai=d["ai"], threshold=d["threshold"], severity=d["severity"],
                                legacy_severity=d["legacy_severity"], status=status)
    if status == "ALERT COUNTDOWN":
        rec.append_transition(status, "ALERT SENT", t=rec.last_t + 10.0)
    return rec.finish()


def test_recomputation_reproduces_a_recorded_real_run(store):
    rec = run_pipeline_recording(store, seed=11, scenario="Severe Accident")
    s = rec.summary
    assert s["ai_result"] is not None and s["severity_max"] is not None and s["threshold_event_detected"] is not None
    cmp_ = rp.compare_recorded_vs_recomputed(store.load(rec.recording_id))
    assert cmp_["identical"], cmp_["mismatches"][:3]
    assert cmp_["decisions_compared"] == 41 and cmp_["source"].startswith(rp.RECOMPUTED_LABEL)


def test_recomputed_decisions_are_labelled_and_leave_the_recording_untouched(store):
    rec = run_pipeline_recording(store, seed=2, scenario="Minor Accident")
    before = copy.deepcopy(rec.to_dict())
    out = rp.recompute_decisions(rec)
    assert len(out) == 41 and all(d["source"] == rp.RECOMPUTED_LABEL for d in out)
    assert rec.to_dict() == before
    assert all(d.get("source") is None for d in rec.decisions)               # recorded ones are not relabelled


def test_recomputation_detects_a_tampered_recording(store):
    rec = run_pipeline_recording(store, seed=5, scenario="Severe Accident")
    tampered = copy.deepcopy(rec.to_dict())
    tampered["decisions"][20]["ai"]["label"] = "Pothole"
    tampered["decisions"][21]["severity"]["score"] = 12.5
    cmp_ = rp.compare_recorded_vs_recomputed(Recording.from_dict(tampered))
    assert not cmp_["identical"]
    fields = {m["field"] for m in cmp_["mismatches"]}
    assert "ai.label" in fields and "severity.score" in fields


def test_recorded_replay_of_a_real_run_matches_what_was_recorded(store):
    rec = run_pipeline_recording(store, seed=7, scenario="Hard Braking")
    frames = rp.ReplaySession(store.load(rec.recording_id)).run_to_end()
    assert len(frames) == 50
    assert [f["decision"]["ai"]["label"] for f in frames if f["decision"]] == [d["ai"]["label"] for d in rec.decisions]
    assert [f["gps"] for f in frames] == [tuple(s["gps"]) for s in rec.samples]


def test_ui_banner_texts_are_present():
    assert rp.REPLAY_BANNER == "SIMULATION REPLAY"
    assert "not real-world accident data" in rp.SYNTHETIC_NOTICE
    assert rp.RECORDED_LABEL != rp.RECOMPUTED_LABEL
