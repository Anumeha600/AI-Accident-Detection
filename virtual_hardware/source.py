"""
VirtualHardwareSensorSource -- the SensorSource the existing pipeline sees in "Virtual Hardware" mode.

    read():  rig.tick()  (vehicle -> sensors -> STM32 -> USART2)
             -> take the telemetry LINE off the virtual UART -> parse it -> reading dict

The reading is built ONLY from the text that came out of the virtual UART. The scenario name selects
what the virtual VEHICLE does; it is never passed to the classifier. Everything downstream
(RollingWindowBuffer, window features, Random Forest, threshold detector, severity_v2, emergency
state machine, recording / replay) is the unchanged Phase 5-10 code.
"""

import random

from common import FEATURE_COLUMNS
from sensor_source import SensorSource, NO_DATA_YET, parse_stm32_line
from virtual_hardware.state import VirtualHardwareRig
from virtual_hardware.clock import TICK_HZ
from virtual_hardware.vehicle import VIRTUAL_SCENARIOS

DEFAULT_RUN_TICKS = 80          # 8 s of simulated time (long enough for a rollover / multi-impact)
TELEMETRY_FIELDS = 12           # 7 firmware fields + vibration, speed, latitude, longitude, gps_fix


def parse_virtual_telemetry(line):
    """
    Parse one virtual telemetry line. The first seven fields go through the SAME parser (and sanity
    limits) the real STM32 source uses; the extension fields follow. Returns
    (reading_dict, gps_dict) or None if the line is malformed.
    """
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="replace")
    parts = line.strip().split(",")
    if len(parts) != TELEMETRY_FIELDS:
        return None
    base = parse_stm32_line(",".join(parts[:7]))
    if base is None:
        return None
    try:
        vibration, speed, lat, lon = (float(p) for p in parts[7:11])
        fix = bool(int(parts[11]))
        timestamp_ms = int(float(parts[0]))
    except ValueError:
        return None
    reading = dict(base)
    reading["vibration_level"] = vibration
    reading["speed_kmh"] = speed
    return reading, {"latitude": lat, "longitude": lon, "fix": fix, "timestamp_ms": timestamp_ms}


class VirtualHardwareSensorSource(SensorSource):
    def __init__(self, seed=None, run_ticks=DEFAULT_RUN_TICKS):
        self._fixed_seed = seed
        self.seed = 0 if seed is None else int(seed)
        self.run_ticks = run_ticks
        self.rig = VirtualHardwareRig(self.seed)
        self._active = False
        self._done = 0
        self.last_error = None
        self.last_reading = None
        self.last_gps = None
        self.last_timestamp_ms = None
        self.last_tick_result = None

    # -- SensorSource interface -------------------------------------------------
    def start(self, scenario="Normal Driving"):
        if scenario not in VIRTUAL_SCENARIOS:
            raise RuntimeError(f"unknown virtual-hardware scenario {scenario!r}")
        self.seed = self._fixed_seed if self._fixed_seed is not None else random.randrange(2 ** 31)
        self.rig.reset(self.seed, scenario)
        self._active = True
        self._done = 0
        self.last_error = None
        self.last_reading = self.last_gps = self.last_timestamp_ms = None

    def read(self):
        if not self._active or self._done >= self.run_ticks:
            return None
        self._done += 1
        self.last_tick_result = self.rig.tick()
        line = self.rig.uart.read_line()                     # what the host really receives
        if line is None:
            return NO_DATA_YET                               # the STM32 sent nothing this period
        parsed = parse_virtual_telemetry(line)
        if parsed is None:
            self.last_error = f"Malformed virtual telemetry line: {line!r}"
            self._active = False
            return None
        reading, gps = parsed
        assert set(reading) == set(FEATURE_COLUMNS), "reading keys drifted from common.FEATURE_COLUMNS"
        self.last_reading, self.last_gps = reading, gps
        self.last_timestamp_ms = gps["timestamp_ms"]
        return reading

    def stop(self):
        self._active = False          # the rig is kept so the lab can keep showing its last state

    # -- extras for the UI / recorder ---------------------------------------------
    @property
    def progress(self):
        return self._done, self.run_ticks

    @property
    def scenario(self):
        return self.rig.scenario

    def set_scenario(self, scenario):
        """Change what the virtual VEHICLE does from now on (live). Does not touch the classifier."""
        self.rig.set_scenario(scenario)

    @property
    def gps_position(self):
        """(lat, lon) from the last telemetry line, or None before the first one."""
        return None if self.last_gps is None else (self.last_gps["latitude"], self.last_gps["longitude"])

    def hardware_info(self):
        """Compact per-sample hardware state stored in recordings (all optional / informational)."""
        snap = self.rig.snapshot()
        return {
            "tick": snap["tick"],
            "saturated": snap["mpu6050"]["saturated"],
            "mcu_event": snap["stm32"]["mcu_event"],
            "vehicle_state": (snap["vehicle"] or {}).get("vehicle_state"),
            "gps_fix": snap["gps"]["fix"],
            "i2c_transactions": snap["i2c"]["count"],
            "uart_packets": snap["uart"]["packets"],
        }

    @property
    def sample_rate_hz(self):
        return TICK_HZ
