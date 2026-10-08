"""
Virtual MPU6050 -- a REGISTER-ORIENTED model of the InvenSense MPU-6050 6-axis IMU, covering the
subset of the register map this project uses. Reads and writes go through the register file, the
same way the STM32 driver (stm32_firmware/mpu6050.c) talks to the real chip over I2C.

REGISTER MAP IMPLEMENTED (addresses/values as in the MPU-6050 register map document)

    0x19  SMPLRT_DIV    R/W  sample rate = gyro_output_rate / (1 + SMPLRT_DIV); reset 0x00
    0x1A  CONFIG        R/W  bits 2:0 DLPF_CFG (0 => gyro output rate 8 kHz, else 1 kHz); reset 0x00
    0x1B  GYRO_CONFIG   R/W  bits 4:3 FS_SEL: 0=+-250, 1=+-500, 2=+-1000, 3=+-2000 deg/s; reset 0x00
    0x1C  ACCEL_CONFIG  R/W  bits 4:3 AFS_SEL: 0=+-2g, 1=+-4g, 2=+-8g, 3=+-16g; reset 0x00
    0x38  INT_ENABLE    R/W  bit 0 DATA_RDY_EN; reset 0x00
    0x3A  INT_STATUS    R    bit 0 DATA_RDY_INT; set when a sample is latched, CLEARED ON READ
    0x3B..0x40  ACCEL_XOUT_H/L, YOUT_H/L, ZOUT_H/L   R   signed 16-bit, big-endian
    0x41..0x42  TEMP_OUT_H/L                         R   T[degC] = raw/340 + 36.53
    0x43..0x48  GYRO_XOUT_H/L, YOUT_H/L, ZOUT_H/L    R   signed 16-bit, big-endian
    0x6B  PWR_MGMT_1    R/W  bit 7 DEVICE_RESET (self-clearing), bit 6 SLEEP; reset 0x40 (asleep)
    0x75  WHO_AM_I      R    always 0x68

Anything else reads 0x00; writes to read-only or unmapped registers are ignored (and counted in
`ignored_writes`), like the silicon. While SLEEP is set the output registers are not updated -- the
driver must write 0x00 to PWR_MGMT_1 first (the virtual STM32 does this at boot).
Multi-byte reads auto-increment the register address, as on the real device.

SATURATION: the full-scale range comes from the CONFIG registers (default +-2 g / +-250 deg/s).
A physical value beyond it is CLIPPED in the output registers, the per-axis saturation flag is
set, and the un-clipped value is kept in `last_physical` for diagnostics. The Phase 7 model was
trained on synthetic signals that exceed these limits (e.g. -9 g impacts), so feeding it clipped
data creates a train/live distribution mismatch. That is intentional in this phase and is NOT
corrected here (no retraining).
"""

from dataclasses import dataclass

I2C_ADDRESS = 0x68
WHO_AM_I_VALUE = 0x68

REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_INT_ENABLE = 0x38
REG_INT_STATUS = 0x3A
REG_ACCEL_XOUT_H = 0x3B
REG_TEMP_OUT_H = 0x41
REG_GYRO_XOUT_H = 0x43
REG_PWR_MGMT_1 = 0x6B
REG_WHO_AM_I = 0x75

REGISTER_NAMES = {
    REG_SMPLRT_DIV: "SMPLRT_DIV", REG_CONFIG: "CONFIG", REG_GYRO_CONFIG: "GYRO_CONFIG",
    REG_ACCEL_CONFIG: "ACCEL_CONFIG", REG_INT_ENABLE: "INT_ENABLE", REG_INT_STATUS: "INT_STATUS",
    REG_PWR_MGMT_1: "PWR_MGMT_1", REG_WHO_AM_I: "WHO_AM_I",
    **{REG_ACCEL_XOUT_H + i: n for i, n in enumerate(
        ["ACCEL_XOUT_H", "ACCEL_XOUT_L", "ACCEL_YOUT_H", "ACCEL_YOUT_L", "ACCEL_ZOUT_H", "ACCEL_ZOUT_L",
         "TEMP_OUT_H", "TEMP_OUT_L",
         "GYRO_XOUT_H", "GYRO_XOUT_L", "GYRO_YOUT_H", "GYRO_YOUT_L", "GYRO_ZOUT_H", "GYRO_ZOUT_L"])},
}

_READ_ONLY = {REG_INT_STATUS, REG_WHO_AM_I, *range(REG_ACCEL_XOUT_H, REG_GYRO_XOUT_H + 6)}
_WRITABLE = {REG_SMPLRT_DIV, REG_CONFIG, REG_GYRO_CONFIG, REG_ACCEL_CONFIG, REG_INT_ENABLE, REG_PWR_MGMT_1}

ACCEL_FULL_SCALE_G = (2.0, 4.0, 8.0, 16.0)
GYRO_FULL_SCALE_DPS = (250.0, 500.0, 1000.0, 2000.0)
ACCEL_LSB_PER_G = (16384.0, 8192.0, 4096.0, 2048.0)
GYRO_LSB_PER_DPS = (131.0, 65.5, 32.8, 16.4)

PWR_SLEEP = 0x40
PWR_RESET = 0x80
INT_DATA_RDY = 0x01

BURST_LENGTH = 14       # ACCEL(6) + TEMP(2) + GYRO(6), read in one I2C transaction


def register_name(address):
    return REGISTER_NAMES.get(address, f"REG_0x{address:02X}")


@dataclass(frozen=True)
class SensorSample:
    accel_x_g: float
    accel_y_g: float
    accel_z_g: float
    gyro_x_dps: float
    gyro_y_dps: float
    gyro_z_dps: float
    temp_c: float = 25.0
    saturated: bool = False


def _s16(high, low):
    v = (high << 8) | low
    return v - 0x10000 if v & 0x8000 else v


def decode_burst(data, accel_range_index=0, gyro_range_index=0):
    """
    Convert a 14-byte burst (ACCEL x3, TEMP, GYRO x3, big-endian int16) to physical units, the way a
    driver does. `saturated` is True when any axis sits at an int16 rail, which is the only
    saturation evidence a real driver can see (the chip itself has no saturation flag).
    """
    if len(data) != BURST_LENGTH:
        raise ValueError(f"expected {BURST_LENGTH} bytes, got {len(data)}")
    a_lsb, g_lsb = ACCEL_LSB_PER_G[accel_range_index], GYRO_LSB_PER_DPS[gyro_range_index]
    ax, ay, az, temp, gx, gy, gz = (_s16(data[i], data[i + 1]) for i in range(0, 14, 2))
    at_rail = any(v in (32767, -32768) for v in (ax, ay, az, gx, gy, gz))
    return SensorSample(ax / a_lsb, ay / a_lsb, az / a_lsb, gx / g_lsb, gy / g_lsb, gz / g_lsb,
                        temp_c=temp / 340.0 + 36.53, saturated=at_rail)


def _to_raw(value, lsb):
    return max(-32768, min(32767, int(round(value * lsb))))


class MPU6050:
    def __init__(self, rng, accel_noise_g=0.04, gyro_noise_dps=1.0, address=I2C_ADDRESS):
        self.address = address
        self._rng = rng
        self.accel_noise_g = accel_noise_g
        self.gyro_noise_dps = gyro_noise_dps
        self._regs = {}
        self.ignored_writes = 0
        self.samples_latched = 0
        self.power_on_reset()

    # -- register file -------------------------------------------------------
    def power_on_reset(self):
        self._regs = {a: 0x00 for a in range(0x00, 0x80)}
        self._regs[REG_PWR_MGMT_1] = PWR_SLEEP
        self._regs[REG_WHO_AM_I] = WHO_AM_I_VALUE
        self._acc_ms = 0.0
        self.saturated_axes = {"ax": False, "ay": False, "az": False, "gx": False, "gy": False, "gz": False}
        self.last_physical = {k: 0.0 for k in self.saturated_axes}
        self._write_output_registers(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 25.0)

    def read_register(self, address):
        if not 0 <= address <= 0xFF:
            raise ValueError(f"register address out of range: {address:#x}")
        value = self._regs.get(address, 0x00)
        if address == REG_INT_STATUS:
            self._regs[REG_INT_STATUS] = 0x00            # clear-on-read
        return value

    def read_registers(self, start_address, length):
        """Burst read with address auto-increment."""
        if length < 1:
            raise ValueError("length must be >= 1")
        return bytes(self.read_register(start_address + i) for i in range(length))

    def write_register(self, address, value):
        if not 0 <= value <= 0xFF:
            raise ValueError(f"register value out of range: {value:#x}")
        if address not in _WRITABLE:
            self.ignored_writes += 1
            return
        if address == REG_PWR_MGMT_1 and value & PWR_RESET:
            self.power_on_reset()                       # DEVICE_RESET restores the defaults (incl. SLEEP)
            return
        self._regs[address] = value

    # -- decoded configuration -----------------------------------------------
    @property
    def sleeping(self):
        return bool(self._regs[REG_PWR_MGMT_1] & PWR_SLEEP)

    @property
    def accel_range_index(self):
        return (self._regs[REG_ACCEL_CONFIG] >> 3) & 0x3

    @property
    def gyro_range_index(self):
        return (self._regs[REG_GYRO_CONFIG] >> 3) & 0x3

    @property
    def accel_full_scale_g(self):
        return ACCEL_FULL_SCALE_G[self.accel_range_index]

    @property
    def gyro_full_scale_dps(self):
        return GYRO_FULL_SCALE_DPS[self.gyro_range_index]

    @property
    def sample_rate_hz(self):
        gyro_rate = 8000.0 if (self._regs[REG_CONFIG] & 0x7) == 0 else 1000.0
        return gyro_rate / (1 + self._regs[REG_SMPLRT_DIV])

    @property
    def saturated(self):
        return any(self.saturated_axes.values())

    # -- the physical side: called by the simulation, not by the STM32 ---------
    def sense(self, accel_g, gyro_dps, temp_c=25.0, dt_ms=100.0):
        """
        Present the true physical quantities to the sensor for dt_ms of simulated time. When a sample
        period has elapsed (and the chip is awake) a noisy, range-limited, quantised sample is
        latched into the output registers and DATA_RDY is raised. Returns True if a sample latched.
        """
        self._acc_ms += dt_ms
        period_ms = 1000.0 / self.sample_rate_hz
        if self._acc_ms + 1e-9 < period_ms or self.sleeping:
            return False
        self._acc_ms = max(0.0, self._acc_ms - period_ms)
        noisy_a = [a + self._rng.normal(0.0, self.accel_noise_g) for a in accel_g]
        noisy_g = [g + self._rng.normal(0.0, self.gyro_noise_dps) for g in gyro_dps]
        self._write_output_registers(*noisy_a, *noisy_g, temp_c)
        self.samples_latched += 1
        self._regs[REG_INT_STATUS] |= INT_DATA_RDY
        return True

    def _write_output_registers(self, ax, ay, az, gx, gy, gz, temp_c):
        a_lsb, g_lsb = ACCEL_LSB_PER_G[self.accel_range_index], GYRO_LSB_PER_DPS[self.gyro_range_index]
        phys = {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}
        self.last_physical = dict(phys)                  # pre-saturation values, for diagnostics
        # saturated == the value does not fit the int16 output register at the configured range
        self.saturated_axes = {k: not -32768 <= round(v * (a_lsb if k[0] == "a" else g_lsb)) <= 32767
                               for k, v in phys.items()}
        raws = [_to_raw(ax, a_lsb), _to_raw(ay, a_lsb), _to_raw(az, a_lsb),
                int(round((temp_c - 36.53) * 340.0)),
                _to_raw(gx, g_lsb), _to_raw(gy, g_lsb), _to_raw(gz, g_lsb)]
        order = [REG_ACCEL_XOUT_H, REG_ACCEL_XOUT_H + 2, REG_ACCEL_XOUT_H + 4, REG_TEMP_OUT_H,
                 REG_GYRO_XOUT_H, REG_GYRO_XOUT_H + 2, REG_GYRO_XOUT_H + 4]
        for reg, raw in zip(order, raws):
            raw16 = raw & 0xFFFF
            self._regs[reg] = raw16 >> 8
            self._regs[reg + 1] = raw16 & 0xFF

    # -- the driver side --------------------------------------------------------
    def decode_burst(self, data):
        """Convert a 14-byte burst (ACCEL, TEMP, GYRO) to physical units using the CURRENT range settings."""
        return decode_burst(data, self.accel_range_index, self.gyro_range_index)

    def read_sensor_sample(self):
        """Read the 14 output registers directly (no bus) and decode them."""
        return self.decode_burst(self.read_registers(REG_ACCEL_XOUT_H, BURST_LENGTH))
