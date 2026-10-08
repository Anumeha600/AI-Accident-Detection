"""
VirtualHardwareRig -- wires the virtual parts together and advances them one 100 ms tick at a time.

    VirtualVehicle -> MPU6050 / VibrationSensor / GPS     (physical side)
    VirtualSTM32 --I2C1--> MPU6050, --USART2--> receiver   (embedded side)
    VirtualSTM32 --USART1--> ESP32/GSM alert unit (stub)

The rig is the only place that knows the whole picture. Nothing here classifies anything: the AI
pipeline sits downstream of USART2 (see source.VirtualHardwareSensorSource).
"""

from dataclasses import dataclass
from typing import Optional

from virtual_hardware import mpu6050 as mpu
from virtual_hardware.clock import VirtualClock, TICK_MS
from virtual_hardware.gps import VirtualGPS
from virtual_hardware.i2c import VirtualI2CBus
from virtual_hardware.stm32 import VirtualSTM32, VirtualAlertUnit
from virtual_hardware.uart import VirtualUART, UARTPacket
from virtual_hardware.vehicle import VirtualVehicle, VehicleState
from virtual_hardware.vibration_sensor import VibrationSensor


@dataclass(frozen=True)
class TickResult:
    vehicle: VehicleState
    packet: Optional[UARTPacket]      # telemetry line the STM32 transmitted this tick (None if none)


class VirtualHardwareRig:
    def __init__(self, seed=0, scenario="Normal Driving"):
        self._build(seed, scenario)

    def _build(self, seed, scenario):
        self.seed = int(seed)
        self.clock = VirtualClock(self.seed)
        self.vehicle = VirtualVehicle(self.clock)
        self.vehicle.set_scenario(scenario)
        self.mpu = mpu.MPU6050(self.clock.rng("mpu6050"))
        self.vibration = VibrationSensor(self.clock.rng("vibration"))
        self.gps = VirtualGPS(self.clock.rng("gps"))
        self.i2c = VirtualI2CBus(self.clock, "I2C1")
        self.i2c.register_device(self.mpu.address, self.mpu)
        self.uart = VirtualUART(self.clock, "USART2")           # telemetry -> host / AI pipeline
        self.uart_alert = VirtualUART(self.clock, "USART1")     # STM32 -> ESP32/GSM alert unit
        self.alert_unit = VirtualAlertUnit()
        self.stm32 = VirtualSTM32(self.clock, self.i2c, self.uart, self.uart_alert,
                                  self.vibration, self.gps, self.alert_unit)
        self.stm32.boot()                                       # "power-on": probe + configure the MPU6050
        self.last_vehicle: Optional[VehicleState] = None
        self.last_packet: Optional[UARTPacket] = None

    def reset(self, seed=None, scenario=None):
        self._build(self.seed if seed is None else seed, scenario or self.vehicle.scenario)

    def set_scenario(self, scenario):
        self.vehicle.set_scenario(scenario)

    @property
    def scenario(self):
        return self.vehicle.scenario

    def tick(self) -> TickResult:
        """Advance 100 ms: world -> sensors -> STM32 sampling cycle -> UART."""
        self.clock.advance()
        vs = self.vehicle.step()
        self.mpu.sense(vs.accel_g, vs.gyro_dps, dt_ms=TICK_MS)
        self.vibration.sense(vs)
        self.gps.sense(vs)
        self.last_vehicle = vs
        self.last_packet = self.stm32.sample_cycle()
        return TickResult(vs, self.last_packet)

    def set_emergency_status(self, status):
        self.stm32.set_emergency_status(status)

    # -- everything the UI needs, as plain data ----------------------------------------
    def snapshot(self):
        vs, s, m = self.last_vehicle, self.stm32.last_sample, self.mpu
        gps = self.gps.report()
        return {
            "tick": self.clock.tick,
            "time": self.clock.timestamp(),
            "seed": self.seed,
            "scenario": self.scenario,
            "vehicle": vs.as_dict() if vs else None,
            "mpu6050": {
                "address": m.address, "who_am_i": self.stm32.who_am_i,
                "accel_range_g": m.accel_full_scale_g, "gyro_range_dps": m.gyro_full_scale_dps,
                "sample_rate_hz": m.sample_rate_hz, "sleeping": m.sleeping,
                "accel": (s.accel_x_g, s.accel_y_g, s.accel_z_g) if s else (0.0, 0.0, 0.0),
                "gyro": (s.gyro_x_dps, s.gyro_y_dps, s.gyro_z_dps) if s else (0.0, 0.0, 0.0),
                "temp_c": s.temp_c if s else None,
                "saturated": m.saturated, "saturated_axes": dict(m.saturated_axes),
                "physical_before_saturation": dict(m.last_physical),
                "samples_latched": m.samples_latched, "ignored_writes": m.ignored_writes,
            },
            "vibration": {"level": self.vibration.level},
            "gps": gps,
            "stm32": {
                "cpu": self.stm32.cpu_state, "i2c1": "ACTIVE" if self.stm32.i2c_active else "IDLE",
                "usart2": "ACTIVE" if self.stm32.usart2_active else "IDLE", "baud": self.uart.baud,
                "sample_rate_hz": self.stm32.sample_rate_hz, "samples": self.stm32.samples,
                "gpio_status": self.stm32.gpio_status, "mcu_event": self.stm32.mcu_event,
                "event_count": self.stm32.event_count, "status_text": self.stm32.status_text,
                "alert_output": self.stm32.alert_output, "buzzer": self.stm32.buzzer, "led": self.stm32.led,
                "i2c_errors": self.stm32.i2c_errors, "fault_reason": self.stm32.fault_reason,
                "uptime_ms": self.stm32.uptime_ms,
            },
            "i2c": {"bus": self.i2c.name, "devices": self.i2c.devices, "count": self.i2c.transaction_count,
                    "recent": [t.summary() for t in self.i2c.recent(8)],
                    "last_phases": list(self.i2c.last().phases) if self.i2c.last() else []},
            "uart": {"name": self.uart.name, "baud": self.uart.baud, "packets": self.uart.packet_count,
                     "recent": [f"{p.time}  {p.text}" for p in self.uart.recent(6)],
                     "wire_time_ms": self.uart.last_packet.wire_time_ms if self.uart.last_packet else 0.0},
            "alert_unit": {"state": self.alert_unit.state, "commands": list(self.alert_unit.commands[-4:])},
        }
