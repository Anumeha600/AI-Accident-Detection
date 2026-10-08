"""
PHASE 10.5 -- Virtual hardware (project-specific functional simulation).

    Virtual Vehicle -> Virtual MPU6050 / vibration sensor / GPS
        -> Virtual I2C1 -> Virtual STM32 -> Virtual UART (USART2)
        -> VirtualHardwareSensorSource -> the EXISTING AI / threshold / severity pipeline

This is a functional model of the project's hardware architecture. It is NOT a
cycle-accurate STM32 emulator and the C firmware in stm32_firmware/ is NOT executed
on a simulated CPU. All values (sensors, GPS, alerts) are SIMULATED; no real SMS/GSM
message is ever sent.
"""

from virtual_hardware.clock import VirtualClock
from virtual_hardware.vehicle import VIRTUAL_SCENARIOS
from virtual_hardware.state import VirtualHardwareRig
from virtual_hardware.source import VirtualHardwareSensorSource

__all__ = ["VirtualClock", "VirtualHardwareRig", "VirtualHardwareSensorSource", "VIRTUAL_SCENARIOS"]
