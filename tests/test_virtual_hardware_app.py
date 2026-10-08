"""
Phase 10.5 through the real Streamlit script (AppTest): the Virtual Hardware data source, the
VIRTUAL HARDWARE LAB tab, pause / step controls, recording, replay and the emergency flow.
Same manual-tick fixture as test_app_smoke.py (one at.run() == one simulation tick).
"""

import os
from pathlib import Path

import pytest

import replay as rp
from recording_store import RecordingStore
from test_app_smoke import click, manual_ticks, new_app, STM32_MODE  # noqa: F401  (manual_ticks is an autouse fixture)
from virtual_hardware import VirtualHardwareSensorSource, VIRTUAL_SCENARIOS

pytestmark = pytest.mark.slow

VIRTUAL_MODE = "Virtual Hardware (Phase 10.5)"
SIM_MODE = "Simulated (Phase 1-4)"


def recordings_dir() -> Path:
    return Path(os.environ["ACCIDENT_RECORDINGS_DIR"])


def virtual_app(scenario="Normal Driving", seed=1):
    at = new_app()
    at.sidebar.radio[0].set_value(VIRTUAL_MODE).run()
    at.sidebar.selectbox[0].set_value(scenario).run()
    at.sidebar.number_input[0].set_value(seed).run()
    return at


def button(at, label):
    return next(b for b in list(at.sidebar.button) + list(at.main.button) if b.label == label)


def tick(at, n=1):
    for _ in range(n):
        at.run()


def test_data_source_selector_offers_all_three_modes_and_defaults_to_simulated():
    at = new_app()
    assert not at.exception
    assert list(at.sidebar.radio[0].options) == [SIM_MODE, VIRTUAL_MODE, STM32_MODE]
    assert at.sidebar.radio[0].value == SIM_MODE
    assert [t.label for t in at.tabs] == ["LIVE SIMULATION", "VIRTUAL HARDWARE LAB", "REPLAY", "RESEARCH LAB"]
    assert list(at.sidebar.selectbox[0].options) == ["Normal Driving", "Pothole", "Hard Braking", "Sharp Turn",
                                                     "Minor Accident", "Severe Accident"]
    assert any("Showing the virtual hardware at power-on" in i.value for i in at.info)


def test_virtual_mode_shows_the_honesty_banner_and_all_eight_scenarios():
    at = new_app()
    at.sidebar.radio[0].set_value(VIRTUAL_MODE).run()
    assert not at.exception and not at.error
    assert list(at.sidebar.selectbox[0].options) == VIRTUAL_SCENARIOS
    assert not any("VIRTUAL HARDWARE" in w.value for w in at.warning)          # no paragraph-sized banner any more
    assert any(c.value == "SIMULATED · SYNTHETIC DATA · NO REAL GSM" for c in at.sidebar.caption)
    md = " ".join(m.value for m in at.markdown)
    assert "VIRTUAL HARDWARE LAB" in md and "VIRTUAL SIMULATION" in md and "SYNTHETIC DATA" in md and "NO REAL GSM" in md
    limits = next(e for e in at.expander if e.label == "Simulation limitations")
    text = " ".join(m.value for m in limits.markdown)
    assert "not a cycle-accurate STM32 emulator" in text and "NO REAL SMS/GSM MESSAGE WAS SENT" in text
    assert "train/live distribution mismatch" in text and "GPS is **SIMULATED**" in text
    assert at.sidebar.header[0].value == "SIMULATION" and len(at.sidebar.caption) <= 2     # short sidebar
    assert [t.label for t in at.tabs] == ["VIRTUAL HARDWARE LAB", "LIVE SIMULATION", "REPLAY", "RESEARCH LAB"]   # the lab is the hero page
    assert list(at.selectbox(key="lab_scenario").options) == VIRTUAL_SCENARIOS and at.selectbox(key="lab_scenario").value == "Normal Driving"
    assert at.sidebar.button[0].label == "Start" and any(b.label == "Pause" for b in at.sidebar.button)
    assert any(b.label == "Step one tick" for b in at.sidebar.button)


def test_virtual_run_drives_the_existing_pipeline_and_the_lab():
    at = virtual_app("Hard Braking")
    click(at, "Start")
    tick(at, 14)
    assert not at.exception and not at.error
    src = at.session_state.sensor_source
    assert isinstance(src, VirtualHardwareSensorSource) and src.rig.stm32.samples == 15
    assert at.session_state.running and src.rig.snapshot()["stm32"]["cpu"] == "RUNNING"
    # the AI got a prediction from the window built out of UART telemetry (never from the scenario label)
    assert at.session_state.prediction_label and at.session_state.last_decision["severity"]["severity"]
    assert len(at.session_state.history["vibration"]) == 15
    assert at.session_state.latest_reading["speed_kmh"] > 0
    # SIMULATED GPS comes from the virtual GPS receiver, not from the random walk
    assert (at.session_state.gps_lat, at.session_state.gps_lon) == pytest.approx(src.gps_position)
    # the lab tab: inspector + logs
    assert [r for r in at.selectbox if r.label == "Inspect component"][0].value == "MPU6050"
    md = " ".join(m.value for m in at.markdown)
    assert "COMPONENT INSPECTOR" in md and "WHO_AM_I" in md and "0x68" in md and "AI EVENT ANALYSIS" in md
    assert "Prediction" in md and "EMERGENCY STATE" in md and ("SYSTEM NORMAL" in md or "EVENT DETECTED" in md)
    assert {e.label for e in at.expander} >= {"UART TELEMETRY", "I²C BUS LOG", "Simulation limitations"}
    codes = " ".join(c.value for c in at.code)
    assert "READ 0x68" in codes and "ACCEL_XOUT_H" in codes and "12.97" in codes
    assert [r for r in at.radio if r.label == "Oscilloscope channel"][0].options == ["ACCEL", "GYRO", "SPEED", "VIBRATION"]


def test_component_inspector_follows_the_selection():
    at = virtual_app()
    click(at, "Start")
    tick(at, 3)
    [r for r in at.selectbox if r.label == "Inspect component"][0].set_value("GPS").run()
    md = " ".join(m.value for m in at.markdown)
    assert "COMPONENT INSPECTOR · GPS" in md and "SIMULATED GPS" in md
    [r for r in at.selectbox if r.label == "Inspect component"][0].set_value("STM32").run()
    md = " ".join(m.value for m in at.markdown)
    assert "COMPONENT INSPECTOR · STM32" in md and "RUNNING" in md and "115200" in md


def test_pause_resume_and_step_one_tick():
    at = virtual_app()
    click(at, "Start")
    tick(at, 4)
    src = at.session_state.sensor_source
    button(at, "Pause").click()
    at.run()
    assert not at.session_state.running and at.session_state.virtual_paused
    assert at.session_state.recorder is not None                       # the recording stays open while paused
    paused_at = src.rig.stm32.samples
    at.run(); at.run()
    assert src.rig.stm32.samples == paused_at                          # nothing advances while paused
    button(at, "Step one tick").click()
    at.run()
    assert src.rig.stm32.samples == paused_at + 1 and at.session_state.virtual_paused
    assert not at.session_state.running
    button(at, "Start").click()                                        # resume the SAME run
    at.run()
    assert at.session_state.running and at.session_state.sensor_source is src
    assert src.rig.stm32.samples > paused_at + 1


def test_step_from_idle_starts_a_paused_run_one_sampling_cycle_at_a_time():
    at = virtual_app()
    button(at, "Step one tick").click()
    at.run()
    src = at.session_state.sensor_source
    assert src.rig.stm32.samples == 1 and src.rig.snapshot()["tick"] == 1
    assert at.session_state.virtual_paused and not at.session_state.running
    button(at, "Step one tick").click()
    at.run()
    assert at.session_state.sensor_source is src and src.rig.stm32.samples == 2
    assert src.last_timestamp_ms == 200


def test_changing_the_scenario_live_changes_the_virtual_vehicle_not_the_classifier():
    at = virtual_app("Normal Driving")
    click(at, "Start")
    tick(at, 3)
    src = at.session_state.sensor_source
    assert src.scenario == "Normal Driving"
    at.sidebar.selectbox[0].set_value("Severe Accident").run()
    assert src.scenario == "Severe Accident" and at.session_state.scenario == "Severe Accident"
    at.selectbox(key="lab_scenario").set_value("Rollover").run()                 # the lab's own dropdown
    assert src.scenario == "Rollover" and at.session_state.scenario == "Rollover"
    assert src.rig.snapshot()["vehicle"]["scenario"] == "Rollover"


def test_accident_reaches_the_emergency_countdown_and_it_can_be_cancelled_from_the_lab():
    at = virtual_app("Severe Accident", seed=1)
    click(at, "Start")
    for _ in range(80):
        if at.session_state.status == "ALERT COUNTDOWN":
            break
        at.run()
    assert at.session_state.status == "ALERT COUNTDOWN"
    at.run()
    src = at.session_state.sensor_source
    snap = src.rig.snapshot()
    assert snap["stm32"]["alert_output"] and snap["stm32"]["buzzer"] and snap["stm32"]["led"] == "RED"
    assert snap["alert_unit"]["state"].startswith("ARMED")
    assert at.session_state.countdown_remaining is not None and at.session_state.severity == "HIGH"
    md = " ".join(m.value for m in at.markdown)
    assert "EMERGENCY COUNTDOWN" in md and "NO REAL SMS/GSM MESSAGE WAS SENT" in md
    assert "ALERT COUNTDOWN" in md                                              # AI panel: System = ALERT COUNTDOWN
    button(at, "Pause").click()      # a new HIGH window would (by design) restart the countdown, so stop the stream
    at.run()
    next(b for b in at.main.button if b.label == "Cancel simulated alert").click()
    at.run()
    assert at.session_state.status == "ALERT CANCELLED"
    at.run()
    assert not src.rig.stm32.alert_output and src.rig.alert_unit.state == "CANCELLED"


def test_virtual_run_is_recorded_and_replayable():
    at = virtual_app("Severe Accident", seed=2)
    click(at, "Start")
    tick(at, 15)
    at.session_state.sensor_source._done = 78                         # skip ahead (AppTest leaks a loop per run)
    for _ in range(10):
        if not at.session_state.running:
            break
        at.run()
    assert not at.session_state.running
    # the alert countdown may still be pending; cancel it so the recording is finalised
    if at.session_state.status == "ALERT COUNTDOWN":
        next(b for b in at.main.button if b.label == "Cancel simulated alert").click()
        at.run()
    at.run()
    rows = RecordingStore(recordings_dir()).list_recordings()
    assert len(rows) == 1 and rows[0]["error"] is None
    rec = RecordingStore(recordings_dir()).load(rows[0]["recording_id"])
    d = rec.data
    assert d["source_type"] == "virtual_hardware" and d["synthetic"] is True and d["scenario"] == "Severe Accident"
    assert d["virtual_hardware"]["seed"] == 2 and "cycle-accurate" in d["virtual_hardware"]["notice"]
    assert all("gps" in s and "hw" in s for s in d["samples"]) and len(rec.decisions) == len(rec.samples) - 9
    assert [s["t"] for s in rec.samples[:3]] == [0.0, 0.1, 0.2]
    assert rec.decisions[-1]["ai"]["label"] and rec.decisions[-1]["severity"]["severity"]
    # replay it in the REPLAY tab
    at.run()
    sel = next(s for s in at.main.selectbox if s.label == "Saved recording")
    assert len(sel.options) == 1
    next(b for b in at.main.button if b.label == "Replay").click()
    at.run()
    assert at.session_state.replay_session.status == rp.PLAYING and not at.exception
    assert "virtual_hardware" in " ".join(str(m.value) for m in at.markdown)


def test_reset_returns_the_virtual_hardware_to_power_on():
    at = virtual_app()
    click(at, "Start")
    tick(at, 5)
    click(at, "Reset")
    src = at.session_state.sensor_source
    assert not at.session_state.running and not at.session_state.virtual_paused
    assert isinstance(src, VirtualHardwareSensorSource) and src.rig.stm32.samples == 0


def test_other_modes_still_work_next_to_virtual_mode():
    at = new_app()
    click(at, "Start")
    tick(at, 12)
    assert not at.exception and at.session_state.prediction_label
    at2 = new_app()
    at2.sidebar.radio[0].set_value(STM32_MODE).run()
    assert not at2.exception and any("STM32 HARDWARE MODE" in w.value for w in at2.warning)
    assert [b.label for b in at2.sidebar.button][:3] == ["Start", "Stop", "Reset"]
