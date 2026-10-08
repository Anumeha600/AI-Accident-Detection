"""
PHASE 5 -- Sensor input abstraction.
--------------------------------------
Defines a common interface (SensorSource) so the rest of the pipeline
(window buffer, feature extraction, model, UI) does not care whether
readings come from the Phase 1-4 simulator or, later, real STM32
hardware over USB/UART.

SimulatedSensorSource reuses Phase 4's dataset_generator.generate_recording()
directly -- it does NOT reimplement the sensor-simulation math -- so a
fresh, randomized 50-sample (5-second) recording is generated each time
a scenario is started, and streamed out ONE READING AT A TIME (like a
live sensor would), instead of being computed and shown all at once.

FUTURE HARDWARE COMPATIBILITY:
    A later STM32SensorSource would implement the same three methods
    (start, read, stop) by opening a serial port and parsing one line
    per call to read() instead of pulling from a generated DataFrame.
    Nothing else in the pipeline (window buffer, predict_expanded.py,
    app.py) would need to change -- they only depend on read() returning
    a dict with the FEATURE_COLUMNS keys.

IMPORTANT: All readings produced here are SIMULATED / SYNTHETIC.
"""

import time
from abc import ABC, abstractmethod

import numpy as np

import dataset_generator as dg
from common import FEATURE_COLUMNS


class _NoDataYet:
    """Type of the NO_DATA_YET sentinel (falsy, prints readably)."""

    def __bool__(self):
        return False

    def __repr__(self):
        return "NO_DATA_YET"


# Returned by a LIVE source's read() when the stream is still healthy but no
# valid sample arrived during this (short, bounded) call. It is NOT the end
# of the stream -- keep calling read(). None still means ended/unavailable.
NO_DATA_YET = _NoDataYet()


class SensorSource(ABC):
    """Common interface for anything that can produce sensor readings."""

    @abstractmethod
    def start(self, scenario: str):
        """Begin producing readings for the given scenario."""

    @abstractmethod
    def read(self):
        """
        Return the next reading as a dict, or None once the stream has
        ended / become unavailable.

        Live sources (STM32SensorSource) may also return the NO_DATA_YET
        sentinel meaning "stream is still alive, no sample arrived within
        this short call" -- see NO_DATA_YET below. SimulatedSensorSource
        never returns it.
        """

    @abstractmethod
    def stop(self):
        """Stop producing readings."""


class SimulatedSensorSource(SensorSource):
    """
    Streams one freshly generated SIMULATED/SYNTHETIC recording at a
    time, one reading per read() call. Reuses Phase 4's
    dataset_generator.generate_recording() so the sensor-simulation
    math lives in exactly one place in the whole project.
    """

    def __init__(self, seed=None):
        # No fixed seed by default -- each Start click produces a
        # genuinely different recording, which is what we want for an
        # interactive demo (dataset_generator.py itself still fixes a
        # seed for reproducible DATASET generation -- a different
        # purpose from this live demo).
        self._rng = np.random.default_rng(seed)
        self._recording = None
        self._index = 0
        self._recording_counter = 0

    def start(self, scenario: str):
        self._recording = dg.generate_recording(self._rng, scenario, self._recording_counter)
        self._recording_counter += 1
        self._index = 0

    def read(self):
        if self._recording is None or self._index >= len(self._recording):
            return None
        row = self._recording.iloc[self._index]
        self._index += 1
        return {col: float(row[col]) for col in FEATURE_COLUMNS}

    @property
    def progress(self):
        """(current_index, total_samples) -- used for a progress bar in the UI."""
        if self._recording is None:
            return 0, 0
        return self._index, len(self._recording)

    def stop(self):
        self._recording = None
        self._index = 0


# ---------------------------------------------------------------------------
# PHASE 6 -- STM32 + MPU6050 real-hardware sensor source
# ---------------------------------------------------------------------------
#
# VERIFICATION STATUS: this Python code has been statically checked
# (py_compile) and exercised against a SIMULATED serial connection (see
# _self_test() below, and common._MockSerial) using deliberately valid
# and malformed sample lines. It has NOT been tested against real
# STM32 hardware in this environment -- no physical board is available
# here. See the Phase 6 write-up for how to test it once you connect
# real hardware.
#
# Expected serial line format, sent by the STM32 firmware in
# stm32_firmware/ once per sample:
#     timestamp_ms,accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps
# Example:
#     1234,0.02,-0.01,1.01,2.3,-1.7,4.2

# Order the STM32 firmware writes fields in (timestamp first).
STM32_SERIAL_FIELD_ORDER = [
    "timestamp_ms",
    "accel_x_g", "accel_y_g", "accel_z_g",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
]
# The subset that are real sensor fields (timestamp_ms is dropped after
# parsing -- it's not one of common.FEATURE_COLUMNS).
STM32_SENSOR_FIELDS = ["accel_x_g", "accel_y_g", "accel_z_g",
                        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]

# Loose sanity bounds used only to reject obviously corrupted lines (a
# torn/garbled serial read, electrical noise, etc.) -- generous relative
# to the MPU6050's configured +/-2g / +/-250 deg/s range. This is
# corruption detection, not a physics constraint.
_ACCEL_SANITY_LIMIT_G = 20.0
_GYRO_SANITY_LIMIT_DPS = 3000.0


def parse_stm32_line(line):
    """
    Parse one line of the STM32 serial format (see module docstring
    above for the exact format and an example).

    Returns a dict with the 6 STM32_SENSOR_FIELDS keys (as floats) on
    success, or None if the line is malformed, incomplete, or has a
    value outside the sanity bounds. Callers should treat None as
    "skip this line and try the next one," not as a fatal error -- a
    single garbled line from a live serial link is expected occasionally
    and is not itself evidence of disconnection.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("ascii", errors="strict")
        except UnicodeDecodeError:
            return None

    line = line.strip()
    if not line:
        return None

    parts = line.split(",")
    if len(parts) != len(STM32_SERIAL_FIELD_ORDER):
        return None

    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None

    reading = dict(zip(STM32_SERIAL_FIELD_ORDER, values))
    del reading["timestamp_ms"]

    for key in ("accel_x_g", "accel_y_g", "accel_z_g"):
        if not (-_ACCEL_SANITY_LIMIT_G <= reading[key] <= _ACCEL_SANITY_LIMIT_G):
            return None
    for key in ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps"):
        if not (-_GYRO_SANITY_LIMIT_DPS <= reading[key] <= _GYRO_SANITY_LIMIT_DPS):
            return None

    return reading


class STM32SensorSource(SensorSource):
    """
    Reads REAL accelerometer/gyroscope data from an STM32 + MPU6050
    board over a serial (USB/UART) connection -- see stm32_firmware/
    for the firmware that produces this data and the exact wire format
    (parse_stm32_line() above).

    PLACEHOLDER FIELDS (Phase 6 scope): the vibration sensor and vehicle
    speed sensing are not wired up to real hardware yet. vibration_level
    and speed_kmh below are filled with fixed PLACEHOLDER constants --
    NOT read from any sensor -- purely so the returned dict still has
    every key common.FEATURE_COLUMNS expects. This is clearly not real
    data for those two fields; do not interpret it as such.

    read() NEVER blocks for long: it waits at most one serial timeout
    (default 0.1 s) per call, so a Streamlit UI calling it once per rerun
    stays responsive. Return values:
        dict         -- a valid reading
        NO_DATA_YET  -- stream still alive, nothing valid arrived this call
        None         -- stream ended/unavailable; last_error says why

    read() returns None ONLY when the connection itself is judged lost: a
    serial exception, MAX_CONSECUTIVE_BAD_LINES unreadable lines in a row
    (counted across calls), or no valid sample for `stall_timeout`
    seconds (silent/unplugged/unpowered board). A single bad line or one
    quiet moment does NOT end the session. last_error always holds the
    human-readable reason after a None.
    """

    VIBRATION_PLACEHOLDER = 0.3   # not from any sensor -- see class docstring
    SPEED_PLACEHOLDER = 0.0       # not from any sensor -- see class docstring
    MAX_CONSECUTIVE_BAD_LINES = 20
    DEFAULT_STALL_TIMEOUT_S = 5.0

    def __init__(self, port, baudrate=115200, timeout=0.1, serial_connection=None,
                 stall_timeout=None):
        """
        port : the serial port name, e.g. "COM5" on Windows (see the
               Phase 6 write-up for how to find yours in Device Manager).
               Not hard-coded -- callers (e.g. app.py's sidebar) supply it.
        serial_connection : inject a pre-built object exposing
               .readline() and .close() instead of opening a real port.
               Used by _self_test() below (a SIMULATED serial test, not
               real hardware) so the parsing/retry logic can be verified
               without any physical board.
        timeout : per-readline() serial timeout in seconds. Kept short
               (0.1 s) so each read() call returns quickly.
        stall_timeout : seconds without any valid sample after which the
               stream is declared dead (read() returns None). Default:
               DEFAULT_STALL_TIMEOUT_S.
        """
        self._port_name = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._injected_connection = serial_connection
        self._connection = None
        self._stall_timeout = self.DEFAULT_STALL_TIMEOUT_S if stall_timeout is None else stall_timeout
        self._consecutive_bad_lines = 0
        self._last_valid_time = None
        self._last_error = None

    def start(self, scenario: str = None):
        # `scenario` exists only for interface compatibility with
        # SensorSource/SimulatedSensorSource -- real hardware doesn't
        # know about scenarios, it just streams whatever it measures.
        self._last_error = None
        self._consecutive_bad_lines = 0
        self._last_valid_time = time.monotonic()

        if self._injected_connection is not None:
            self._connection = self._injected_connection
            return

        try:
            import serial  # pyserial -- imported lazily so the rest of
                            # the app works without it installed unless
                            # STM32SensorSource is actually used.
        except ImportError as exc:
            self._last_error = "pyserial is not installed. Run: pip install pyserial"
            raise RuntimeError(self._last_error) from exc

        try:
            self._connection = serial.Serial(self._port_name, self._baudrate, timeout=self._timeout)
        except Exception as exc:  # serial.SerialException, OSError, ...
            self._last_error = f"Could not open serial port {self._port_name!r}: {exc}"
            self._connection = None
            raise RuntimeError(self._last_error) from exc

    def _fail(self, message):
        """Record why the stream ended, release the port, signal end (None)."""
        self._last_error = message
        self.stop()
        return None

    def read(self):
        if self._connection is None:
            if self._last_error is None:
                self._last_error = "STM32 sensor source is not started (or was stopped)."
            return None

        # At most ONE empty (timed-out) readline per call, so this blocks
        # for roughly `timeout` seconds -- never the old ~20 s.
        while True:
            if time.monotonic() - self._last_valid_time > self._stall_timeout:
                return self._fail(
                    f"No valid data from the STM32 on {self._port_name} for "
                    f"{self._stall_timeout:g} s -- check the USB/serial cable, "
                    "board power, COM port, baud rate and that the firmware is running."
                )

            try:
                raw_line = self._connection.readline()
            except Exception as exc:  # serial.SerialException, OSError, ...
                return self._fail(
                    f"Serial read failed on {self._port_name} (disconnected?): {exc}"
                )

            if not raw_line:
                # Timed out with no data. The stream may still be alive;
                # the stall check above decides when "quiet" becomes "dead".
                return NO_DATA_YET

            parsed = parse_stm32_line(raw_line)
            if parsed is None:
                self._consecutive_bad_lines += 1
                if self._consecutive_bad_lines >= self.MAX_CONSECUTIVE_BAD_LINES:
                    return self._fail(
                        f"Gave up after {self.MAX_CONSECUTIVE_BAD_LINES} consecutive "
                        "unreadable/malformed lines -- check baud rate, wiring and firmware."
                    )
                continue

            self._consecutive_bad_lines = 0
            self._last_valid_time = time.monotonic()
            reading = dict(parsed)
            reading["vibration_level"] = self.VIBRATION_PLACEHOLDER
            reading["speed_kmh"] = self.SPEED_PLACEHOLDER
            assert set(reading) == set(FEATURE_COLUMNS), "reading keys drifted from common.FEATURE_COLUMNS"
            return reading

    @property
    def last_error(self):
        """Human-readable reason the last read()/start() failed, or None."""
        return self._last_error

    def stop(self):
        if self._connection is not None and hasattr(self._connection, "close"):
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None


class _MockSerial:
    """
    A minimal stand-in for pyserial.Serial, used only by the SIMULATED
    serial test below (and reusable for future automated tests):
    readline() returns the next queued line, exactly like a real serial
    port would deliver it (as bytes).
    """

    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def close(self):
        pass


def _self_test():
    """
    SIMULATED SERIAL TEST -- no hardware involved. Feeds
    STM32SensorSource a mix of valid and deliberately malformed lines
    through a mock serial connection, and prints what it parses. This
    verifies the parsing/retry logic in isolation; it does NOT verify
    the real STM32 firmware or a real serial link. Run directly:
        python sensor_source.py
    """
    sample_lines = [
        b"1234,0.02,-0.01,1.01,2.3,-1.7,4.2\r\n",   # valid
        b"garbage,not,a,valid,line\r\n",             # malformed: wrong field count
        b"1334,0.03,-0.02,0.99,2.1,-1.5,4.0\r\n",    # valid
        b"\r\n",                                      # blank line
        b"1434,0.01,9999,1.00,2.0,-1.6,4.1\r\n",     # malformed: accel_y_g out of sanity range
        b"1534,0.02,-0.01,1.02,2.2,-1.8,4.3\r\n",    # valid
    ]
    print("=" * 70)
    print("SIMULATED SERIAL TEST for STM32SensorSource (NO real hardware used)")
    print("=" * 70)

    source = STM32SensorSource(port="MOCK", serial_connection=_MockSerial(sample_lines),
                               stall_timeout=0.2)
    source.start()

    count = 0
    while True:
        reading = source.read()
        if reading is None:
            break
        if reading is NO_DATA_YET:
            time.sleep(0.05)  # the mock has no timeout of its own; real serial does
            continue
        count += 1
        print(f"reading {count}: {reading}")

    print(f"\nParsed {count} valid reading(s) out of {len(sample_lines)} lines fed in "
          "(malformed/blank lines were skipped, not treated as fatal).")
    print("(The final None is the stall timeout firing after the mock ran out of lines.)")
    print(f"last_error after run: {source.last_error}")
    source.stop()


if __name__ == "__main__":
    _self_test()
