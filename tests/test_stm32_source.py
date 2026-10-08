"""
STM32SensorSource behaviour using fake serial connections (NO hardware):
interface contract, responsiveness (no long blocking), and error reporting.
"""

import time

import pytest

import sensor_source
from common import FEATURE_COLUMNS
from sensor_source import (
    NO_DATA_YET, STM32SensorSource, SimulatedSensorSource, SensorSource, _MockSerial,
)

VALID = b"1234,0.02,-0.01,1.01,2.3,-1.7,4.2\r\n"
BAD = b"garbage\r\n"


class QuietSerial:
    """Like a real port with nothing to say: readline() waits `timeout`, returns b''."""

    def __init__(self, timeout=0.05):
        self.timeout = timeout
        self.closed = False

    def readline(self):
        time.sleep(self.timeout)
        return b""

    def close(self):
        self.closed = True


class DyingSerial:
    """Delivers `lines`, then raises like a pulled USB cable."""

    def __init__(self, lines, exc=None):
        self._lines = list(lines)
        self._exc = exc or OSError("device disconnected")
        self.closed = False

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        raise self._exc

    def close(self):
        self.closed = True


def make(conn, **kw):
    s = STM32SensorSource(port="FAKE", serial_connection=conn, **kw)
    s.start()
    return s


# --- interface contract ---------------------------------------------------

def test_is_a_sensor_source_with_start_read_stop():
    s = STM32SensorSource(port="FAKE", serial_connection=_MockSerial([]))
    assert isinstance(s, SensorSource)
    for name in ("start", "read", "stop"):
        assert callable(getattr(s, name))


def test_valid_reading_has_exactly_the_feature_columns():
    s = make(_MockSerial([VALID]))
    r = s.read()
    assert set(r) == set(FEATURE_COLUMNS)
    assert r["vibration_level"] == STM32SensorSource.VIBRATION_PLACEHOLDER
    assert r["speed_kmh"] == STM32SensorSource.SPEED_PLACEHOLDER
    assert r["accel_z_g"] == 1.01


def test_no_data_yet_sentinel_is_falsy_and_distinct_from_none():
    assert NO_DATA_YET is not None
    assert not NO_DATA_YET
    assert repr(NO_DATA_YET) == "NO_DATA_YET"


def test_read_before_start_returns_none_with_reason():
    s = STM32SensorSource(port="FAKE", serial_connection=_MockSerial([VALID]))
    assert s.read() is None
    assert s.last_error


def test_read_after_stop_returns_none():
    s = make(_MockSerial([VALID, VALID]))
    s.stop()
    assert s.read() is None


def test_stop_closes_connection():
    conn = QuietSerial()
    s = make(conn)
    s.stop()
    assert conn.closed


# --- bad lines ------------------------------------------------------------

def test_single_bad_line_is_skipped_not_fatal():
    s = make(_MockSerial([BAD, b"\r\n", VALID]))
    assert s.read() is not None
    assert s.last_error is None


def test_too_many_consecutive_bad_lines_is_reported():
    n = STM32SensorSource.MAX_CONSECUTIVE_BAD_LINES
    s = make(_MockSerial([BAD] * n))
    assert s.read() is None
    assert "consecutive" in s.last_error


def test_bad_line_counter_spans_read_calls_and_resets_on_valid_line():
    n = STM32SensorSource.MAX_CONSECUTIVE_BAD_LINES
    # n-1 bad, a valid line (resets the counter), then n-1 bad again: must NOT give up.
    s = make(_MockSerial([BAD] * (n - 1) + [VALID] + [BAD] * (n - 1) + [VALID]))
    assert s.read() is not None
    assert s.read() is not None
    assert s.last_error is None


# --- responsiveness (the old code could block ~20 s in one call) ------------

def test_read_on_quiet_port_returns_promptly_with_no_data_yet():
    s = make(QuietSerial(timeout=0.05))
    t0 = time.monotonic()
    result = s.read()
    elapsed = time.monotonic() - t0
    assert result is NO_DATA_YET
    assert elapsed < 0.5          # old behaviour: ~20 timeouts inside one call
    assert s.last_error is None   # quiet != dead


def test_default_serial_timeout_is_short():
    assert STM32SensorSource(port="FAKE")._timeout <= 0.2


def test_silent_board_is_eventually_reported_as_stalled():
    s = make(QuietSerial(timeout=0.02), stall_timeout=0.15)
    t0 = time.monotonic()
    result = NO_DATA_YET
    while result is NO_DATA_YET and time.monotonic() - t0 < 5:
        result = s.read()
    assert result is None
    assert "No valid data" in s.last_error
    assert "FAKE" in s.last_error


# --- disconnect / error reporting --------------------------------------------

def test_serial_exception_mid_stream_is_reported_and_port_released():
    conn = DyingSerial([VALID, VALID])
    s = make(conn)
    assert s.read() is not None
    assert s.read() is not None
    assert s.read() is None
    assert "disconnected" in s.last_error.lower()
    assert conn.closed
    assert s.read() is None   # stays ended, error preserved
    assert s.last_error


def test_start_with_missing_port_raises_runtime_error_and_sets_last_error(monkeypatch):
    serial = pytest.importorskip("serial")

    def boom(*a, **k):
        raise serial.SerialException("could not open port 'COM_NOPE'")

    monkeypatch.setattr(serial, "Serial", boom)
    s = STM32SensorSource(port="COM_NOPE")
    with pytest.raises(RuntimeError, match="COM_NOPE"):
        s.start()
    assert "Could not open serial port" in s.last_error
    assert s.read() is None


def test_start_clears_previous_error():
    s = make(DyingSerial([]))
    assert s.read() is None and s.last_error
    s._injected_connection = _MockSerial([VALID])
    s.start()
    assert s.last_error is None
    assert s.read() is not None


# --- simulated source contract -----------------------------------------------

def test_simulated_source_streams_one_recording_then_none_without_error():
    src = SimulatedSensorSource(seed=1)
    src.start("Normal Driving")
    readings = []
    while (r := src.read()) is not None:
        readings.append(r)
    assert len(readings) == 50
    assert all(set(r) == set(FEATURE_COLUMNS) for r in readings)
    assert src.progress == (50, 50)
    # normal completion is distinguishable from a hardware failure:
    assert getattr(src, "last_error", None) is None
    assert src.read() is None   # stays exhausted


def test_simulated_source_never_returns_the_no_data_sentinel():
    src = SimulatedSensorSource(seed=2)
    src.start("Pothole")
    assert not any(src.read() is NO_DATA_YET for _ in range(60))


def test_module_self_test_runs(capsys):
    sensor_source._self_test()
    assert "Parsed 3 valid reading(s) out of 6 lines" in capsys.readouterr().out
