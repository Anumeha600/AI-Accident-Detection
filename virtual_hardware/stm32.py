"""
Virtual STM32 -- a PROJECT-SPECIFIC FUNCTIONAL model of the controller's role in this system.

NOT a cycle-accurate emulator, and the C firmware in stm32_firmware/ is NOT executed on a simulated
Cortex-M CPU. This class re-implements, in Python, what that firmware does at the hardware boundary:

    boot:      read WHO_AM_I over I2C1, wake the MPU6050, configure 10 Hz / +-2 g / +-250 deg/s
    every 100 ms (sampling loop):
               INT_STATUS + 14-byte burst read of the MPU6050 over I2C1,
               read the vibration sensor (ADC1) and the GPS receiver (USART3),
               format one telemetry line and send it on USART2 (115200 baud)
    on request: drive the alert output (LED / buzzer) and send a command line to the ESP32/GSM unit
               (USART1). The ESP32/GSM unit is a stub: it only changes a status string, no SMS/GSM
               message is ever sent.

Telemetry line (CRLF-terminated). The first seven fields are exactly the real firmware's format
(`%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f`); the virtual build appends the fields the real firmware does
not provide yet, so the pipeline receives the full 8-feature contract:

    timestamp_ms,accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps,
    vibration_level,speed_kmh,latitude,longitude,gps_fix

`mcu_event` is a simple on-MCU motion pre-filter (like the MPU6050 motion interrupt). It is NOT the
AI classifier and does not decide the alert.
"""

import math

from virtual_hardware.i2c import I2CError
from virtual_hardware import mpu6050 as mpu

SAMPLE_RATE_HZ = 10
EVENT_ACCEL_DEVIATION_G = 0.8     # | |a| - 1 g | above this => motion event
EVENT_GYRO_DPS = 120.0
EVENT_HOLD_TICKS = 5              # keep the EVENT flag visible for 0.5 s

CPU_RESET, CPU_INIT, CPU_RUNNING, CPU_FAULT = "RESET", "INIT", "RUNNING", "FAULT"

# status strings of the existing emergency state machine (realtime_pipeline.next_status / app.py)
EMERGENCY_ACTIVE = ("ALERT COUNTDOWN", "ALERT SENT")


class VirtualAlertUnit:
    """ESP32/GSM stub: parses STM32 command lines, changes a status string. NEVER sends anything."""

    def __init__(self):
        self.state = "IDLE"
        self.commands = []

    def poll(self, uart):
        while True:
            line = uart.read_line()
            if line is None:
                return
            cmd = line.decode("ascii").strip()
            self.commands.append(cmd)
            self.commands = self.commands[-20:]
            self.state = {"ALERT,COUNTDOWN": "ARMED (countdown)",
                          "ALERT,SEND": "SIMULATED ALERT SENT",
                          "ALERT,CANCEL": "CANCELLED",
                          "ALERT,CLEAR": "IDLE"}.get(cmd, self.state)


class VirtualSTM32:
    def __init__(self, clock, i2c, uart_telemetry, uart_alert, vibration, gps, alert_unit,
                 mpu_address=mpu.I2C_ADDRESS):
        self._clock = clock
        self._i2c = i2c
        self._uart = uart_telemetry
        self._uart_alert = uart_alert
        self._vib = vibration
        self._gps = gps
        self.alert_unit = alert_unit
        self._mpu_addr = mpu_address
        self.reset()

    def reset(self):
        self.cpu_state = CPU_RESET
        self.samples = 0
        self.sample_rate_hz = SAMPLE_RATE_HZ
        self.i2c_errors = 0
        self.who_am_i = None
        self.fault_reason = None
        self.i2c_active = False
        self.usart2_active = False
        self.mcu_event = False
        self.event_count = 0
        self._event_hold = 0
        self.saturation_seen = False
        self.alert_output = False
        self.buzzer = False
        self.emergency_status = "MONITORING"
        self.last_sample = None
        self.last_frame = None
        self._accel_idx = 0
        self._gyro_idx = 0

    # -- state shown in the UI -------------------------------------------------
    @property
    def led(self):
        if self.alert_output:
            return "RED"
        if self.mcu_event or self.emergency_status in ("EVENT DETECTED", "ALERT CANCELLED"):
            return "YELLOW"
        return "GREEN"

    @property
    def gpio_status(self):
        return {"RUNNING": "OK", "INIT": "INIT", "RESET": "RESET", "FAULT": "FAULT"}[self.cpu_state]

    @property
    def uptime_ms(self):
        return self._clock.time_ms

    # -- boot ------------------------------------------------------------------
    def boot(self):
        """Probe and configure the MPU6050 over I2C1 (mirrors MPU6050_Init in stm32_firmware/mpu6050.c)."""
        self.cpu_state = CPU_INIT
        try:
            self.who_am_i = self._i2c.read(self._mpu_addr, mpu.REG_WHO_AM_I, 1)[0]
            if self.who_am_i != mpu.WHO_AM_I_VALUE:
                return self._fault(f"WHO_AM_I = 0x{self.who_am_i:02X}, expected 0x{mpu.WHO_AM_I_VALUE:02X}")
            self._i2c.write(self._mpu_addr, mpu.REG_PWR_MGMT_1, 0x00)       # wake up
            self._i2c.write(self._mpu_addr, mpu.REG_SMPLRT_DIV, 99)         # 1 kHz / 100 = 10 Hz
            self._i2c.write(self._mpu_addr, mpu.REG_CONFIG, 0x03)           # DLPF ~44 Hz
            self._i2c.write(self._mpu_addr, mpu.REG_GYRO_CONFIG, 0x00)      # +-250 deg/s
            self._i2c.write(self._mpu_addr, mpu.REG_ACCEL_CONFIG, 0x00)     # +-2 g
            self._accel_idx = self._gyro_idx = 0
        except I2CError as exc:
            return self._fault(str(exc))
        self.cpu_state = CPU_RUNNING
        return True

    def _fault(self, reason):
        self.cpu_state = CPU_FAULT
        self.fault_reason = reason
        return False

    # -- sampling loop (one call == one 100 ms period) ------------------------------
    def sample_cycle(self):
        """Poll the sensors and transmit one telemetry line. Returns the UART packet or None."""
        self.i2c_active = self.usart2_active = False
        if self.cpu_state == CPU_RESET:
            before = self._i2c.transaction_count
            self.boot()
            self.i2c_active = self._i2c.transaction_count > before
        if self.cpu_state != CPU_RUNNING:
            return None

        before = self._i2c.transaction_count
        try:
            status = self._i2c.read(self._mpu_addr, mpu.REG_INT_STATUS, 1)[0]
            if not status & mpu.INT_DATA_RDY:
                self.i2c_active = True
                return None                                                 # nothing new this period
            burst = self._i2c.read(self._mpu_addr, mpu.REG_ACCEL_XOUT_H, mpu.BURST_LENGTH)
        except I2CError as exc:
            self.i2c_errors += 1
            self.i2c_active = self._i2c.transaction_count > before
            self.fault_reason = str(exc)
            return None
        self.i2c_active = True

        sample = mpu.decode_burst(burst, self._accel_idx, self._gyro_idx)
        vibration = self._vib.level
        gps = self._gps.report()
        self.last_sample = sample
        self.saturation_seen = self.saturation_seen or sample.saturated
        self._update_event(sample)

        frame = (f"{self._clock.time_ms},{sample.accel_x_g:.3f},{sample.accel_y_g:.3f},{sample.accel_z_g:.3f},"
                 f"{sample.gyro_x_dps:.2f},{sample.gyro_y_dps:.2f},{sample.gyro_z_dps:.2f},"
                 f"{vibration:.3f},{gps['speed_kmh']:.2f},{gps['latitude']:.6f},{gps['longitude']:.6f},"
                 f"{int(gps['fix'])}\r\n")
        self.last_frame = frame
        packet = self._uart.transmit(frame)
        self.usart2_active = True
        self.samples += 1
        return packet

    def _update_event(self, s):
        a = math.sqrt(s.accel_x_g ** 2 + s.accel_y_g ** 2 + s.accel_z_g ** 2)
        g = math.sqrt(s.gyro_x_dps ** 2 + s.gyro_y_dps ** 2 + s.gyro_z_dps ** 2)
        if abs(a - 1.0) > EVENT_ACCEL_DEVIATION_G or g > EVENT_GYRO_DPS:
            if not self.mcu_event:
                self.event_count += 1
            self.mcu_event = True
            self._event_hold = EVENT_HOLD_TICKS
        elif self._event_hold > 0:
            self._event_hold -= 1
        else:
            self.mcu_event = False

    # -- emergency output ----------------------------------------------------------
    def set_emergency_status(self, status):
        """Drive the alert output from the emergency state machine's status; tell the ESP32 stub on a change."""
        if status == self.emergency_status:
            return
        previous, self.emergency_status = self.emergency_status, status
        self.alert_output = self.buzzer = status in EMERGENCY_ACTIVE
        command = {"ALERT COUNTDOWN": "ALERT,COUNTDOWN", "ALERT SENT": "ALERT,SEND",
                   "ALERT CANCELLED": "ALERT,CANCEL"}.get(status)
        if command is None and previous in EMERGENCY_ACTIVE + ("ALERT CANCELLED",):
            command = "ALERT,CLEAR"
        if command:
            self._uart_alert.transmit(command + "\r\n")
            self.alert_unit.poll(self._uart_alert)

    @property
    def status_text(self):
        if self.alert_output:
            return "ALERT OUTPUT ACTIVE"
        if self.mcu_event:
            return "EVENT DETECTED"
        return "NO EVENT"
