"""
PHASE 5 -- Real-time software simulator / dashboard.
--------------------------------------------------------------------------
SIMULATED SENSOR STREAM -> 1-second window buffer -> feature extraction
-> Phase 4 ML model -> event classification -> severity lookup ->
emergency-response simulation -> UI.

This dashboard does NOT connect to any real hardware. Every sensor
reading, GPS coordinate, and emergency notification shown here is
SIMULATED / SYNTHETIC (see sensor_source.py, realtime_pipeline.py).

It reuses, without modifying:
    - dataset_generator.py  (Phase 4 recording simulation math)
    - window_features.py    (Phase 3/4 windowing + feature extraction)
    - train_expanded_model.py's EXPANDED_MODEL_PATH constant
    - accident_classifier_model_expanded.joblib (Phase 4's trained model,
      loaded read-only -- never retrained or modified here)

RESEARCH INTEGRITY:
    - This model was trained and evaluated ONLY on simulated data (see
      Phase 2-4). It has NOT been validated on real-world accident data
      and must not be treated as a certified accident-detection system.
    - Severity (NORMAL/LOW/MEDIUM/HIGH) is a fixed, hand-authored lookup
      from the predicted class (see realtime_pipeline.SEVERITY_BY_LABEL)
      -- it is NOT predicted by the ML model.
    - Whatever the model predicts is shown as-is, even if unexpected;
      predictions are never adjusted to make the UI look better.
"""

import os
import time
from pathlib import Path

# The model, threshold/severity configs and the recordings folder are referenced by project-relative paths
# (e.g. "accident_classifier_model_expanded.joblib"). Pin the working directory to this file's folder so the app
# behaves the same however it is launched (local `streamlit run`, another cwd, Streamlit Community Cloud).
os.chdir(Path(__file__).resolve().parent)

import numpy as np
import pandas as pd
import streamlit as st

from common import FEATURE_COLUMNS
from window_features import WINDOW_SIZE
from sensor_source import SimulatedSensorSource, STM32SensorSource, NO_DATA_YET
from predict_expanded import EXPANDED_MODEL_PATH
import severity_v2
import threshold_detector
import replay_ui
from recording_store import RecordingStore, RecordingError, describe_models
from replay import decide_window
from dataset_generator import SAMPLE_RATE_HZ
from virtual_hardware import VirtualHardwareSensorSource, VIRTUAL_SCENARIOS
from virtual_hardware import lab_ui
from realtime_pipeline import (
    RollingWindowBuffer,
    severity_for_label,
    next_status,
    simulate_gps_step,
    accel_magnitude,
    gyro_magnitude,
    GPS_ORIGIN,
)

SCENARIOS = [
    "Normal Driving", "Pothole", "Hard Braking",
    "Sharp Turn", "Minor Accident", "Severe Accident",
]
TICK_DELAY_SECONDS = 0.5  # wall-clock time between UI updates (simulated mode)
# STM32 mode: the board streams ~10 Hz and STM32SensorSource.read() already
# waits at most one short serial timeout, so no extra delay -- otherwise the
# UI would consume fewer samples than arrive and the serial buffer would lag.
STM32_TICK_DELAY_SECONDS = 0.0
VIRTUAL_TICK_DELAY_SECONDS = 0.1  # Virtual Hardware mode: one 100 ms simulation tick per UI refresh
HISTORY_LIMIT = 60        # points kept for the live charts

SOURCE_SIMULATED = "Simulated (Phase 1-4)"
SOURCE_VIRTUAL = "Virtual Hardware (Phase 10.5)"
SOURCE_STM32 = "STM32 (Serial, real accel/gyro)"
SOURCE_MODES = [SOURCE_SIMULATED, SOURCE_VIRTUAL, SOURCE_STM32]
EMPTY_HISTORY = {"tick": [], "accel_mag": [], "gyro_mag": [], "speed": [], "vibration": []}

st.set_page_config(page_title="Accident Detection Simulator (Phase 5)", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    defaults = dict(
        running=False,
        scenario="Normal Driving",
        sensor_source=SimulatedSensorSource(),
        buffer=RollingWindowBuffer(),
        status="MONITORING",
        severity="NORMAL",
        prediction_label=None,
        prediction_confidence=None,
        prediction_probs={},
        countdown_total=10,
        alert_start_time=None,
        countdown_remaining=None,
        gps_lat=GPS_ORIGIN[0],
        gps_lon=GPS_ORIGIN[1],
        gps_rng=np.random.default_rng(),
        history={k: [] for k in EMPTY_HISTORY},
        latest_reading=None,
        recording_progress=(0, 0),
        source_mode=SOURCE_SIMULATED,
        com_port="COM3",
        source_error=None,   # human-readable reason a hardware stream failed
        source_waiting=False,  # STM32 alive but no sample arrived yet
        recorder=None,         # Phase 10: in-progress recording of the current run
        recorder_status="MONITORING",
        record_runs=True,
        last_saved_recording=None,
        record_error=None,
        virtual_paused=False,      # Phase 10.5: a paused virtual-hardware run keeps its source (and recording) open
        virtual_seed=0,            # 0 = pick a random seed at every Start
        last_decision=None,        # latest decide_window() result (AI + threshold + severity_v2)
        lab_idle_source=None,      # power-on rig shown in the lab while another data source is selected
    )
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def reset_run_state(keep_gps=False):
    st.session_state.buffer.reset()
    st.session_state.status = "MONITORING"
    st.session_state.severity = "NORMAL"
    st.session_state.prediction_label = None
    st.session_state.prediction_confidence = None
    st.session_state.prediction_probs = {}
    st.session_state.countdown_remaining = None
    st.session_state.alert_start_time = None
    st.session_state.history = {k: [] for k in EMPTY_HISTORY}
    st.session_state.last_decision = None
    st.session_state.virtual_paused = False
    st.session_state.latest_reading = None
    st.session_state.recording_progress = (0, 0)
    st.session_state.source_error = None
    st.session_state.source_waiting = False
    if not keep_gps:
        st.session_state.gps_lat, st.session_state.gps_lon = GPS_ORIGIN


# ---------------------------------------------------------------------------
# Phase 10: recording of runs (saved to recordings/, replayed in the REPLAY tab)
# ---------------------------------------------------------------------------

recording_store = RecordingStore()


def start_recorder():
    """Begin recording the run that was just started (no-op if recording is switched off)."""
    if not st.session_state.record_runs:
        return
    try:
        st.session_state.recorder = recording_store.start_recording(
            scenario=None if is_stm32_mode else st.session_state.scenario,
            source_type="stm32" if is_stm32_mode else ("virtual_hardware" if is_virtual_mode else "simulated"),
            sample_rate_hz=SAMPLE_RATE_HZ,
            models=describe_models(EXPANDED_MODEL_PATH, threshold_detector.DEFAULT_CONFIG_PATH,
                                   severity_v2.DEFAULT_CONFIG_PATH),
            alert={"countdown_total_s": int(st.session_state.countdown_total)},
        )
        if is_virtual_mode:
            src = st.session_state.sensor_source
            st.session_state.recorder.set_virtual_hardware({
                "seed": src.seed, "initial_scenario": st.session_state.scenario,
                "mpu6050": {"i2c_address": "0x68", "accel_range_g": 2, "gyro_range_dps": 250, "sample_rate_hz": 10},
                "notice": "VIRTUAL HARDWARE SIMULATION - functional model, not a cycle-accurate STM32 emulator. "
                          "Simulated GPS. No real SMS/GSM message was sent.",
            })
        st.session_state.recorder_status = st.session_state.status
    except RecordingError as exc:
        st.session_state.recorder = None
        st.session_state.record_error = f"Recording disabled for this run: {exc}"


def log_status_change():
    rec = st.session_state.recorder
    if rec is not None and st.session_state.status != st.session_state.recorder_status:
        rec.append_transition(st.session_state.recorder_status, st.session_state.status)
        st.session_state.recorder_status = st.session_state.status


def finalize_recorder():
    """Save and close the in-progress recording (empty ones are dropped)."""
    rec = st.session_state.recorder
    if rec is None:
        return
    log_status_change()
    st.session_state.recorder = None
    if rec.n_samples == 0:
        rec.discard()
        return
    try:
        st.session_state.last_saved_recording = rec.finish(save=True).recording_id
    except RecordingError as exc:
        st.session_state.record_error = f"Could not save the recording: {exc}"


def sync_recorder():
    """Log status changes; close the recording once the stream is over AND no alert countdown is pending."""
    if st.session_state.recorder is None:
        return
    log_status_change()
    if (not st.session_state.running and not st.session_state.virtual_paused
            and st.session_state.status != "ALERT COUNTDOWN"):
        finalize_recorder()


# ---------------------------------------------------------------------------
# Sidebar: scenario control
# ---------------------------------------------------------------------------

st.sidebar.header("SIMULATION")
source_mode = st.sidebar.radio(
    "Data source",
    SOURCE_MODES,
    index=SOURCE_MODES.index(st.session_state.source_mode),
    disabled=st.session_state.running or st.session_state.virtual_paused,
)
st.session_state.source_mode = source_mode
is_stm32_mode = source_mode == SOURCE_STM32
is_virtual_mode = source_mode == SOURCE_VIRTUAL
scenario_options = VIRTUAL_SCENARIOS if is_virtual_mode else SCENARIOS
if (is_virtual_mode and not st.session_state.running and not st.session_state.virtual_paused
        and not isinstance(st.session_state.sensor_source, VirtualHardwareSensorSource)):
    st.session_state.sensor_source = VirtualHardwareSensorSource()      # idle, powered-on virtual rig

if is_stm32_mode:
    st.session_state.com_port = st.sidebar.text_input(
        "STM32 serial port (e.g. COM5)",
        value=st.session_state.com_port,
        disabled=st.session_state.running,
    )
    st.sidebar.caption(
        "Phase 6: accelerometer/gyroscope come from REAL STM32+MPU6050 "
        "hardware over serial. Vibration and speed are still PLACEHOLDER "
        "values -- those sensors are not wired up yet."
    )

if st.session_state.scenario not in scenario_options:
    st.session_state.scenario = scenario_options[0]
scenario = st.sidebar.selectbox(
    "Scenario", scenario_options,
    index=scenario_options.index(st.session_state.scenario),
    # a virtual-hardware scenario can be switched LIVE (it changes what the virtual vehicle does)
    disabled=(st.session_state.running and not is_virtual_mode) or is_stm32_mode,
)
st.session_state.scenario = scenario
if is_stm32_mode:
    st.sidebar.caption("Scenario selection is ignored in STM32 mode -- real hardware streams whatever it measures.")
if is_virtual_mode:
    st.session_state.virtual_seed = st.sidebar.number_input(
        "Seed (0 = random)", min_value=0, max_value=2**31 - 1, value=int(st.session_state.virtual_seed),
        disabled=st.session_state.running or st.session_state.virtual_paused,
    )
    live_source = st.session_state.sensor_source
    if isinstance(live_source, VirtualHardwareSensorSource) and live_source.scenario != scenario:
        live_source.set_scenario(scenario)

st.session_state.countdown_total = st.sidebar.number_input(
    "Emergency alert countdown (seconds)",
    min_value=3, max_value=30, value=st.session_state.countdown_total,
    disabled=st.session_state.running,
)

st.session_state.record_runs = st.sidebar.checkbox(
    "Record this run", value=st.session_state.record_runs,
    disabled=st.session_state.running,
)

c1, c2, c3 = st.sidebar.columns(3)
start_clicked = c1.button("Start", disabled=st.session_state.running)
# Virtual Hardware: "Pause" keeps the run (and its recording) open so it can be resumed or stepped.
stop_clicked = c2.button("Pause" if is_virtual_mode else "Stop", disabled=not st.session_state.running)
reset_clicked = c3.button("Reset")
step_clicked = is_virtual_mode and st.sidebar.button(
    "Step one tick", disabled=st.session_state.running,
    help="Advance the virtual hardware by exactly one 100 ms sampling cycle.")


def begin_run():
    """Start a fresh run of the selected data source (and its recording)."""
    finalize_recorder()          # a previous run may still be waiting for its alert countdown
    st.session_state.record_error = None
    reset_run_state(keep_gps=True)
    if is_stm32_mode:
        st.session_state.sensor_source = STM32SensorSource(port=st.session_state.com_port)
    elif is_virtual_mode:
        seed = int(st.session_state.virtual_seed)
        st.session_state.sensor_source = VirtualHardwareSensorSource(seed=seed if seed else None)
    else:
        st.session_state.sensor_source = SimulatedSensorSource()
    try:
        st.session_state.sensor_source.start(scenario)
        st.session_state.running = True
        start_recorder()
        return True
    except RuntimeError as exc:
        st.session_state.running = False
        st.session_state.source_error = f"Could not start sensor source: {exc}"
        return False


do_step = False
if start_clicked:
    if is_virtual_mode and st.session_state.virtual_paused:
        st.session_state.virtual_paused = False        # resume the paused run
        st.session_state.running = True
    else:
        begin_run()

if step_clicked:
    if not st.session_state.virtual_paused and begin_run():
        st.session_state.running = False               # a step run starts paused
        st.session_state.virtual_paused = True
    do_step = st.session_state.virtual_paused

if stop_clicked:
    st.session_state.running = False
    if is_virtual_mode:
        st.session_state.virtual_paused = True
    else:
        st.session_state.sensor_source.stop()

if reset_clicked:
    st.session_state.running = False
    st.session_state.virtual_paused = False
    st.session_state.sensor_source.stop()
    if is_virtual_mode:
        st.session_state.sensor_source = VirtualHardwareSensorSource()    # back to the power-on state
    reset_run_state(keep_gps=False)

if is_stm32_mode:
    st.sidebar.caption(
        "Accelerometer/gyroscope: REAL (STM32+MPU6050 over serial). "
        "Vibration/speed: PLACEHOLDER (not yet wired). GPS and the "
        "emergency alert remain SIMULATED."
    )
elif is_virtual_mode:
    st.sidebar.caption("SIMULATED · SYNTHETIC DATA · NO REAL GSM")
else:
    st.sidebar.caption(
        "All sensor values, GPS coordinates, and emergency alerts in this "
        "dashboard are SIMULATED / SYNTHETIC. No real hardware is connected."
    )
if not is_virtual_mode:
    st.sidebar.caption(f"Model file (read-only): {EXPANDED_MODEL_PATH}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

if not is_virtual_mode:      # virtual mode: the lab header carries the title, badges and limitations
    st.title("AI-Based Smart Accident Detection -- Real-Time Software Simulator")
if is_virtual_mode:
    pass
elif is_stm32_mode:
    st.warning(
        "STM32 HARDWARE MODE: accelerometer/gyroscope readings come from REAL "
        "STM32+MPU6050 hardware over serial. Vibration and vehicle speed are "
        "still PLACEHOLDER values (not yet wired to real sensors). GPS and the "
        "emergency alert below remain SIMULATED and no real message is sent. "
        "The underlying model was trained and evaluated ONLY on simulated data "
        "(Phases 2-4) and has NOT been validated against real-world accident "
        "data -- predictions on real sensor input here are exploratory only."
    )
else:
    st.warning(
        "SIMULATED DEMO ONLY. Sensor data, GPS, and the emergency alert below are "
        "all synthetic/simulated -- no real hardware is connected and no real "
        "message is sent. The underlying model was trained and evaluated ONLY on "
        "simulated data (Phases 2-4) and has NOT been validated against "
        "real-world accident data."
    )

# ---------------------------------------------------------------------------
# One simulation tick: pull a reading, update buffer/prediction/status/gps
# ---------------------------------------------------------------------------

if st.session_state.running or do_step:
    reading = st.session_state.sensor_source.read()
    if reading is None:
        # None = stream ended/unavailable. Normal end of a simulated
        # recording and a hardware/serial failure are different things:
        # only the latter sets last_error.
        st.session_state.running = False
        st.session_state.virtual_paused = False
        st.session_state.source_waiting = False
        hw_error = getattr(st.session_state.sensor_source, "last_error", None)
        if is_stm32_mode:
            st.session_state.source_error = (
                hw_error or "STM32 stream ended unexpectedly (no error detail reported)."
            )
        elif hw_error:
            st.session_state.source_error = hw_error
        st.session_state.sensor_source.stop()
    elif reading is NO_DATA_YET:
        # Stream alive, no sample this tick -- keep running and retry.
        st.session_state.source_waiting = True
    else:
        st.session_state.source_waiting = False
        st.session_state.latest_reading = reading
        # STM32SensorSource has no fixed-length recording, hence no .progress
        st.session_state.recording_progress = getattr(st.session_state.sensor_source, "progress", (0, 0))
        st.session_state.buffer.push(reading)

        tick = len(st.session_state.history["tick"])
        hist = st.session_state.history
        hist["tick"].append(tick)
        hist["accel_mag"].append(accel_magnitude(reading))
        hist["gyro_mag"].append(gyro_magnitude(reading))
        hist["speed"].append(reading["speed_kmh"])
        hist["vibration"].append(reading["vibration_level"])
        for key in hist:
            hist[key] = hist[key][-HISTORY_LIMIT:]

        virtual_source = (st.session_state.sensor_source
                          if isinstance(st.session_state.sensor_source, VirtualHardwareSensorSource) else None)
        if virtual_source is not None and virtual_source.gps_position is not None:
            # SIMULATED GPS from the virtual GPS receiver (arrived through the UART telemetry line)
            st.session_state.gps_lat, st.session_state.gps_lon = virtual_source.gps_position
        else:
            st.session_state.gps_lat, st.session_state.gps_lon = simulate_gps_step(
                st.session_state.gps_rng, st.session_state.gps_lat, st.session_state.gps_lon
            )
        if st.session_state.recorder is not None:
            st.session_state.recorder.append_sample(
                reading, gps=(st.session_state.gps_lat, st.session_state.gps_lon),
                hardware=virtual_source.hardware_info() if virtual_source is not None else None)

        # Rolling 1-second window: predict on every tick once the buffer
        # is full, using the last WINDOW_SIZE readings.
        if st.session_state.buffer.is_full():
            # One shared call: AI classification (+ threshold detector and severity_v2 for the
            # recording). The AI part is exactly the Phase 4 predict_window_event() as before.
            decision = decide_window(st.session_state.buffer.as_list())
            label, confidence, probs = (decision["ai"]["label"], decision["ai"]["confidence"],
                                        decision["ai"]["probabilities"])
            st.session_state.last_decision = decision
            st.session_state.prediction_label = label
            st.session_state.prediction_confidence = confidence
            st.session_state.prediction_probs = probs
            st.session_state.severity = severity_for_label(label)

            new_status = next_status(st.session_state.status, st.session_state.severity)
            if new_status == "ALERT COUNTDOWN" and st.session_state.status != "ALERT COUNTDOWN":
                st.session_state.alert_start_time = time.time()
                st.session_state.countdown_remaining = st.session_state.countdown_total
            st.session_state.status = new_status
            if st.session_state.recorder is not None:
                st.session_state.recorder.append_decision(
                    ai=decision["ai"], threshold=decision["threshold"], severity=decision["severity"],
                    legacy_severity=decision["legacy_severity"], status=new_status)
                log_status_change()

# Advance the emergency countdown independently of the sensor stream: a
# real emergency timer must keep counting down even after the simulated
# recording itself has ended (e.g. a short 5-second recording with the
# event near the end, or a stopped/destroyed vehicle sending no more
# readings) -- it must not freeze just because sensor data stopped.
if st.session_state.status == "ALERT COUNTDOWN" and st.session_state.alert_start_time is not None:
    elapsed = time.time() - st.session_state.alert_start_time
    remaining = st.session_state.countdown_total - int(elapsed)
    st.session_state.countdown_remaining = remaining
    if remaining <= 0:
        st.session_state.status = "ALERT SENT"
        st.session_state.countdown_remaining = None

sync_recorder()

# ---------------------------------------------------------------------------
# Sensor-stream status banner (persists across reruns until Start/Reset)
# ---------------------------------------------------------------------------

if st.session_state.source_error:
    st.error(
        f"SENSOR SOURCE PROBLEM: {st.session_state.source_error}  "
        "Fix the connection and press Start to try again."
    )
elif st.session_state.running and st.session_state.source_waiting:
    st.warning(f"Connected, waiting for data from the STM32 on {st.session_state.com_port}...")
elif (not st.session_state.running and st.session_state.recording_progress[1]
      and st.session_state.recording_progress[0] >= st.session_state.recording_progress[1]):
    st.info("Recording finished. Press Start to generate a new simulated recording.")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

if is_virtual_mode:      # the hardware lab is the hero page in virtual mode
    tab_lab, tab_live, tab_replay, tab_research = st.tabs(
        ["VIRTUAL HARDWARE LAB", "LIVE SIMULATION", "REPLAY", "RESEARCH LAB"])
else:
    tab_live, tab_lab, tab_replay, tab_research = st.tabs(
        ["LIVE SIMULATION", "VIRTUAL HARDWARE LAB", "REPLAY", "RESEARCH LAB"])
left, right = tab_live.columns([2, 1])

with left:
    st.subheader("Live Sensor Data (SIMULATED)")
    reading = st.session_state.latest_reading or {c: 0.0 for c in FEATURE_COLUMNS}
    r1 = st.columns(4)
    r1[0].metric("Accel X (g)", f"{reading['accel_x_g']:.3f}")
    r1[1].metric("Accel Y (g)", f"{reading['accel_y_g']:.3f}")
    r1[2].metric("Accel Z (g)", f"{reading['accel_z_g']:.3f}")
    r1[3].metric("Vibration", f"{reading['vibration_level']:.3f}")
    r2 = st.columns(4)
    r2[0].metric("Gyro X (deg/s)", f"{reading['gyro_x_dps']:.2f}")
    r2[1].metric("Gyro Y (deg/s)", f"{reading['gyro_y_dps']:.2f}")
    r2[2].metric("Gyro Z (deg/s)", f"{reading['gyro_z_dps']:.2f}")
    r2[3].metric("Speed (km/h)", f"{reading['speed_kmh']:.1f}")

    idx, total = st.session_state.recording_progress
    if total:
        st.progress(idx / total, text=f"Recording progress: sample {idx}/{total}")

    st.subheader("Sensor Visualization")
    hist = st.session_state.history
    if hist["tick"]:
        motion_df = pd.DataFrame(
            {
                "Acceleration magnitude (g)": hist["accel_mag"],
                "Gyro magnitude (deg/s)": hist["gyro_mag"],
            },
            index=hist["tick"],
        )
        st.line_chart(motion_df)
        speed_df = pd.DataFrame({"Speed (km/h)": hist["speed"]}, index=hist["tick"])
        st.line_chart(speed_df)
    else:
        st.caption("Press Start to see live charts.")

with right:
    st.subheader("AI Classification")
    st.caption(
        "Phase 4 window-based Random Forest "
        f"({EXPANDED_MODEL_PATH}), trained/evaluated ONLY on simulated data."
    )
    if st.session_state.prediction_label:
        st.markdown(f"**Event:** {st.session_state.prediction_label.upper()}")
        st.markdown(f"**Confidence:** {st.session_state.prediction_confidence:.1%}")
        probs_df = pd.DataFrame.from_dict(
            st.session_state.prediction_probs, orient="index", columns=["Probability"]
        ).sort_values("Probability", ascending=False)
        st.bar_chart(probs_df)
    else:
        st.caption(f"Collecting the first {WINDOW_SIZE}-sample (1-second) window...")

    st.subheader("Severity")
    st.caption(
        "Fixed rule-based lookup from the predicted event class -- NOT "
        "predicted by the ML model (see realtime_pipeline.SEVERITY_BY_LABEL)."
    )
    severity = st.session_state.severity
    if severity == "HIGH":
        st.error(f"Severity: {severity}")
    elif severity == "MEDIUM":
        st.warning(f"Severity: {severity}")
    elif severity == "LOW":
        st.info(f"Severity: {severity}")
    else:
        st.success(f"Severity: {severity}")

    st.subheader("Event Status")
    status = st.session_state.status
    st.markdown(f"### {status}")

    if status == "ALERT COUNTDOWN":
        st.error(f"ACCIDENT DETECTED -- Severity: {st.session_state.severity}")
        st.markdown(f"## Emergency notification will be sent in: {st.session_state.countdown_remaining}")
        if st.button("CANCEL ALERT"):
            st.session_state.status = "ALERT CANCELLED"
            st.session_state.countdown_remaining = None
            st.session_state.alert_start_time = None
    elif status == "ALERT SENT":
        st.error("EMERGENCY ALERT SENT -- SIMULATED (no real SMS/GSM message was sent)")
    elif status == "ALERT CANCELLED":
        st.warning("ALERT CANCELLED by user.")
    elif status == "EVENT DETECTED":
        st.info("EVENT DETECTED -- monitoring, below the alert threshold.")

    st.subheader("GPS: SIMULATED")
    st.caption("No real GPS hardware connected. Coordinates are a simulated random walk from an arbitrary starting point.")
    st.write(f"Latitude: {st.session_state.gps_lat:.6f}")
    st.write(f"Longitude: {st.session_state.gps_lon:.6f}")

sync_recorder()   # also catches a status change made by the CANCEL ALERT button above

if st.session_state.record_error:
    st.sidebar.caption(f"Recording: {st.session_state.record_error}")
elif st.session_state.last_saved_recording:
    st.sidebar.caption(f"Last run recorded: {st.session_state.last_saved_recording}")



def set_lab_scenario(name):
    """Lab scenario button: change what the virtual vehicle does (never the classifier)."""
    st.session_state.scenario = name
    src = st.session_state.sensor_source
    if isinstance(src, VirtualHardwareSensorSource):
        src.set_scenario(name)


def cancel_alert():
    st.session_state.status = "ALERT CANCELLED"
    st.session_state.countdown_remaining = None
    st.session_state.alert_start_time = None


with tab_lab:
    if is_virtual_mode:
        lab_source = st.session_state.sensor_source
    else:
        if st.session_state.lab_idle_source is None:
            st.session_state.lab_idle_source = VirtualHardwareSensorSource(seed=1)
        lab_source = st.session_state.lab_idle_source
    # Drive the alert output (LED / buzzer / ESP32 stub) from the EXISTING emergency state machine.
    if is_virtual_mode:
        lab_source.rig.set_emergency_status(st.session_state.status)
    lab_decision = st.session_state.last_decision if is_virtual_mode else None
    lab_ui.render_lab(
        lab_source, is_virtual_mode, st.session_state.scenario,
        running=st.session_state.running, paused=st.session_state.virtual_paused,
        pipeline={
            "label": st.session_state.prediction_label if is_virtual_mode else None,
            "confidence": st.session_state.prediction_confidence if is_virtual_mode else None,
            "severity": (f"{lab_decision['severity']['severity']} ({lab_decision['severity']['score']:.0f})"
                         if lab_decision and lab_decision.get("severity") else
                         (st.session_state.severity if is_virtual_mode else None)),
            "status": st.session_state.status if is_virtual_mode else None,
            "countdown": st.session_state.countdown_remaining,
            "threshold": (("ACCIDENT" if lab_decision["threshold"]["event_detected"] else "NO EVENT")
                          if lab_decision and lab_decision.get("threshold") else None),
            "run_state": ("RUNNING" if st.session_state.running else "PAUSED" if st.session_state.virtual_paused
                          else "FINISHED" if lab_source.progress[0] else "IDLE"),
        },
        history=st.session_state.history, on_scenario=set_lab_scenario, on_cancel=cancel_alert)

with tab_research:
    # Phase 11: reads finished experiment result files; runs an experiment only when its button is pressed
    from experiments import research_ui
    from experiments.experiment_runner import run_experiment, load_config as load_experiment_config
    research_ui.render(st, run_experiment=run_experiment, load_config=load_experiment_config)

with tab_replay:
    replay_ui.render(recording_store)   # may st.rerun() itself while a replay is playing

# ---------------------------------------------------------------------------
# Drive the tick loop. Keep auto-refreshing while the sensor stream is
# still live OR while an alert countdown is in progress -- a real
# emergency timer must keep counting even after the simulated recording
# itself has ended (see the countdown block above).
# ---------------------------------------------------------------------------

if st.session_state.running or st.session_state.status == "ALERT COUNTDOWN":
    time.sleep(STM32_TICK_DELAY_SECONDS if is_stm32_mode
               else VIRTUAL_TICK_DELAY_SECONDS if is_virtual_mode else TICK_DELAY_SECONDS)
    st.rerun()
