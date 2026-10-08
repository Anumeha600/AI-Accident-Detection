"""
Smoke tests that run the real Streamlit script headlessly (streamlit's
AppTest). Serial hardware is faked.

app.py drives itself with `time.sleep(); st.rerun()`. Inside AppTest every
st.rerun() costs seconds, so the fixture below turns both into no-ops and the
tests advance the app one tick at a time with at.run() (one run == one tick).
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.slow

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

STM32_MODE = "STM32 (Serial, real accel/gyro)"
VALID = b"1234,0.02,-0.01,1.01,2.3,-1.7,4.2\r\n"


@pytest.fixture(autouse=True)
def manual_ticks(monkeypatch):
    real_sleep = time.sleep

    def sleep_unless_from_app(seconds):
        # Only skip app.py's own pacing sleeps; AppTest/streamlit internals
        # poll with time.sleep and must keep really sleeping.
        if not sys._getframe(1).f_code.co_filename.endswith("app.py"):
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep_unless_from_app)

    # On some Windows hosts asyncio's loopback socketpair() intermittently
    # fails with WinError 10013 when AppTest creates its per-run event loop.
    # That is an environment flake unrelated to app.py, so retry just that call.
    real_new_event_loop = asyncio.new_event_loop

    def retrying_new_event_loop():
        for attempt in range(40):
            try:
                return real_new_event_loop()
            except PermissionError:
                if attempt == 39:
                    raise
                real_sleep(0.2)

    monkeypatch.setattr(asyncio, "new_event_loop", retrying_new_event_loop)
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)


def new_app():
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def click(at, label):
    next(b for b in at.sidebar.button if b.label == label).click()
    at.run()


def tick_until_stopped(at, max_ticks=200):
    for _ in range(max_ticks):
        if not at.session_state.running:
            return
        at.run()
    raise AssertionError("app never stopped running")


def all_error_text(at):
    return " ".join(e.value for e in at.error)


def select_stm32(at):
    at.sidebar.radio[0].set_value(STM32_MODE).run()


def test_app_starts_cleanly_in_default_simulated_mode():
    at = new_app()
    assert not at.exception
    assert not at.error
    assert "Real-Time Software Simulator" in at.title[0].value


def test_simulated_run_finishes_normally_without_hardware_error():
    at = new_app()
    click(at, "Start")
    assert at.session_state.running is True
    for _ in range(12):                      # fills the 10-sample window -> a prediction is made
        at.run()
    assert at.session_state.prediction_label is not None
    # AppTest leaks an event loop per run() and this host's socket limits are hit
    # after ~40 runs, so fast-forward the stream rather than tick all 50 samples.
    at.session_state.sensor_source._index = 47
    tick_until_stopped(at)
    assert not at.exception
    assert at.session_state.recording_progress == (50, 50)
    assert at.session_state.source_error is None
    assert "SENSOR SOURCE PROBLEM" not in all_error_text(at)
    assert any("Recording finished" in i.value for i in at.info)


def test_stm32_mode_with_unopenable_port_shows_the_real_problem(monkeypatch):
    serial = pytest.importorskip("serial")

    def boom(*a, **k):
        raise serial.SerialException("could not open port 'COM_NOPE': file not found")

    monkeypatch.setattr(serial, "Serial", boom)
    at = new_app()
    select_stm32(at)
    at.sidebar.text_input[0].set_value("COM_NOPE").run()
    click(at, "Start")
    assert not at.exception
    assert at.session_state.running is False
    text = all_error_text(at)
    assert "SENSOR SOURCE PROBLEM" in text and "COM_NOPE" in text
    assert not any("Recording finished" in i.value for i in at.info)


def test_stm32_disconnect_mid_stream_is_reported_not_called_recording_finished(monkeypatch):
    serial = pytest.importorskip("serial")

    class UnpluggedAfterFew:
        def __init__(self, *a, **k):
            self.n = 0

        def readline(self):
            self.n += 1
            if self.n <= 3:
                return VALID
            raise serial.SerialException("ClearCommError failed (PermissionError(13, 'Access is denied.'))")

        def close(self):
            pass

    monkeypatch.setattr(serial, "Serial", UnpluggedAfterFew)
    at = new_app()
    select_stm32(at)
    click(at, "Start")
    tick_until_stopped(at)
    assert not at.exception           # also guards the (fixed) missing-.progress crash on real readings
    text = all_error_text(at)
    assert "SENSOR SOURCE PROBLEM" in text
    assert "Serial read failed" in text and "Access is denied" in text
    assert not any("Recording finished" in i.value for i in at.info)
    assert at.session_state.latest_reading is not None      # the good samples were consumed
    assert len(at.session_state.history["tick"]) == 3


def test_stm32_silent_board_shows_waiting_then_stall_error(monkeypatch):
    serial = pytest.importorskip("serial")

    class Silent:
        def __init__(self, *a, **k):
            pass

        def readline(self):
            return b""

        def close(self):
            pass

    monkeypatch.setattr(serial, "Serial", Silent)
    at = new_app()
    select_stm32(at)
    click(at, "Start")
    at.run()
    assert at.session_state.running is True              # quiet != ended
    assert at.session_state.source_waiting is True
    assert any("waiting for data" in w.value for w in at.warning)

    at.session_state.sensor_source._stall_timeout = 0.0   # simulate the stall window elapsing
    at.run()
    assert at.session_state.running is False
    assert "No valid data from the STM32" in all_error_text(at)


def test_countdown_expires_to_alert_sent():
    at = new_app()
    at.session_state.status = "ALERT COUNTDOWN"
    at.session_state.countdown_total = 10
    at.session_state.alert_start_time = time.time() - 11
    at.run()
    assert at.session_state.status == "ALERT SENT"
    assert any("EMERGENCY ALERT SENT" in e.value for e in at.error)


def test_countdown_in_progress_shows_remaining_seconds_and_cancel_works():
    at = new_app()
    at.session_state.status = "ALERT COUNTDOWN"
    at.session_state.countdown_total = 10
    at.session_state.alert_start_time = time.time() - 3
    at.run()
    assert at.session_state.status == "ALERT COUNTDOWN"
    assert 6 <= at.session_state.countdown_remaining <= 7
    next(b for b in at.main.button if b.label == "CANCEL ALERT").click()
    at.run()
    assert at.session_state.status == "ALERT CANCELLED"
