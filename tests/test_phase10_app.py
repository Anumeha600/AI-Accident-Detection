"""
Phase 10 through the real Streamlit script (AppTest): a live run is recorded, the REPLAY tab lists /
plays / resets it, corrupt files do not break the app. Serial hardware is not involved.
Uses the same manual-tick fixture as test_app_smoke.py (st.rerun and app.py's sleeps are neutralised).
"""

import json
import os
import time
from pathlib import Path

import pytest

import replay as rp
from recording_store import RecordingStore
from test_app_smoke import click, manual_ticks, new_app, tick_until_stopped  # noqa: F401  (fixture re-used)

pytestmark = pytest.mark.slow


def recordings_dir() -> Path:
    return Path(os.environ["ACCIDENT_RECORDINGS_DIR"])


def run_live(at, alert_sent=False):
    """Start a simulated run, tick until the 10-sample window is full, fast-forward, finish."""
    click(at, "Start")
    for _ in range(12):
        at.run()
    at.session_state.sensor_source._index = 47          # AppTest leaks a loop per run(); skip ahead
    if alert_sent:
        at.session_state.status = "ALERT COUNTDOWN"
        at.session_state.countdown_total = 10
        at.session_state.alert_start_time = time.time() - 11
    tick_until_stopped(at)
    at.run()


def replay_buttons(at):
    return {b.label: b for b in at.main.button}


def test_live_run_is_recorded_and_saved_with_everything_needed_for_replay():
    at = new_app()
    run_live(at)
    assert not at.exception
    rows = RecordingStore(recordings_dir()).list_recordings()
    assert len(rows) == 1 and rows[0]["error"] is None
    rec = RecordingStore(recordings_dir()).load(rows[0]["recording_id"])
    assert rec.scenario == "Normal Driving" and rec.data["source_type"] == "simulated" and rec.data["synthetic"]
    # fast-forwarding in run_live() skips samples, so the count is "ticks actually run", not 50
    n = len(rec.samples)
    assert n >= 12 and len(rec.decisions) == n - 9 and rec.summary["duration_s"] == pytest.approx(n / 10)
    d = rec.decisions[-1]
    assert d["ai"]["label"] and 0 <= d["ai"]["confidence"] <= 1 and d["ai"]["probabilities"]
    assert d["threshold"]["classification"] in ("ACCIDENT", "NORMAL") and d["severity"]["severity"]
    assert all("gps" in s for s in rec.samples) and rec.data["alert"]["countdown_total_s"] == 10
    assert set(rec.data["models"]) == {"classifier", "threshold_config", "severity_config"}
    assert at.session_state.last_saved_recording == rec.recording_id
    assert at.session_state.recorder is None


def test_alert_that_fires_after_the_stream_ends_is_part_of_the_recording():
    at = new_app()
    run_live(at, alert_sent=True)
    (row,) = RecordingStore(recordings_dir()).list_recordings()
    rec = RecordingStore(recordings_dir()).load(row["recording_id"])
    assert [t["to"] for t in rec.transitions][-1] == "ALERT SENT"
    assert rec.summary["alert_outcome"] == "ALERT SENT"
    assert rp.ReplaySession(rec).state_at(len(rec.samples) - 1)["status"] == "ALERT SENT"


def test_cancelled_alert_is_recorded():
    at = new_app()
    click(at, "Start")
    for _ in range(3):
        at.run()
    at.session_state.sensor_source._index = 47
    at.session_state.status = "ALERT COUNTDOWN"
    at.session_state.alert_start_time = time.time()
    tick_until_stopped(at)
    assert at.session_state.recorder is not None                    # still open: countdown pending
    next(b for b in at.main.button if b.label == "CANCEL ALERT").click()
    at.run()
    (row,) = RecordingStore(recordings_dir()).list_recordings()
    assert row["alert_outcome"] == "ALERT CANCELLED"


def test_recording_can_be_switched_off():
    at = new_app()
    at.sidebar.checkbox[0].uncheck().run()
    run_live(at)
    assert RecordingStore(recordings_dir()).list_recordings() == []
    assert not at.exception


def test_each_run_gets_its_own_recording():
    at = new_app()
    run_live(at)
    run_live(at)
    rows = RecordingStore(recordings_dir()).list_recordings()
    assert len(rows) == 2 and rows[0]["recording_id"] != rows[1]["recording_id"]


def test_replay_tab_with_no_recordings_is_graceful():
    at = new_app()
    assert not at.exception and not at.error
    assert any("SIMULATION REPLAY" in w.value and "not real-world accident data" in w.value for w in at.warning)
    assert any("No saved recordings yet" in i.value for i in at.info)


def test_replay_tab_lists_loads_and_plays_a_saved_run():
    at = new_app()
    run_live(at)
    at.run()
    assert not at.exception
    assert any("SIMULATION REPLAY" in w.value for w in at.warning)
    sel = next(s for s in at.main.selectbox if s.label == "Saved recording")
    assert len(sel.options) == 1
    sess = at.session_state.replay_session
    n = sess.n_samples
    assert n >= 12 and sess.status == rp.READY

    replay_buttons(at)["Replay"].click()
    at.run()
    assert at.session_state.replay_session.status == rp.PLAYING
    assert at.session_state.replay_session.position >= 1
    replay_buttons(at)["Pause"].click()
    at.run()
    paused_at = at.session_state.replay_session.position
    assert at.session_state.replay_session.status == rp.PAUSED
    replay_buttons(at)["Step"].click()
    at.run()
    assert at.session_state.replay_session.position == paused_at + 1
    replay_buttons(at)["Reset"].click()
    at.run()
    assert at.session_state.replay_session.position == 0 and at.session_state.replay_session.status == rp.READY
    assert any("RECORDED RESULT" in m.value for m in at.markdown) or any(
        "RECORDED RESULT" in str(getattr(s, "value", "")) for s in at.subheader)


def test_replay_runs_to_completion_and_stop_keeps_position():
    at = new_app()
    run_live(at)
    at.run()
    replay_buttons(at)["Replay"].click()
    at.run()
    n = at.session_state.replay_session.n_samples
    at.session_state.replay_session.position = n - 3                 # near the end
    at.run()
    at.run()
    sess = at.session_state.replay_session
    assert sess.status == rp.FINISHED and sess.position == n
    assert not at.exception
    assert any("Emergency state (recorded)" in str(e.value) for e in list(at.info) + list(at.warning) + list(at.error))


def test_corrupt_recording_files_do_not_break_the_app():
    at = new_app()
    run_live(at)
    d = recordings_dir()
    (d / "garbage.json").write_text("{this is not json")
    (d / "wrong-version.json").write_text(json.dumps({"schema_version": 99, "recording_id": "wrong-version"}))
    at.run()
    assert not at.exception
    assert any("unreadable recording" in e.label for e in at.expander)
    assert len(next(s for s in at.main.selectbox if s.label == "Saved recording").options) == 1


def test_only_corrupt_recordings_shows_info_not_a_crash():
    at = new_app()
    (recordings_dir()).mkdir(parents=True, exist_ok=True)
    (recordings_dir() / "bad.json").write_text("[]")
    at.run()
    assert not at.exception and any("No saved recordings yet" in i.value for i in at.info)


def test_recompute_panel_is_separate_and_reports_identical():
    at = new_app()
    run_live(at)
    at.run()
    replay_buttons(at)["Recompute now"].click()
    at.run()
    cmp_ = at.session_state.replay_compare
    assert cmp_["identical"] and cmp_["source"].startswith("LIVE RECOMPUTATION")
    assert at.session_state.replay_session.position == 0           # recomputation did not touch the replay
