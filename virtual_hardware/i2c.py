"""
Virtual I2C bus (master = the virtual STM32, devices = register-file models such as the MPU6050).

Supports device registration, register read / write transactions, a NACK for unknown addresses
and a bounded transaction log. Each logged transaction also records the wire-level phases, e.g. for
a register read:  START, ADDR 0x68 W, REG 0x75, RESTART, ADDR 0x68 R, DATA 0x68, NACK, STOP.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

from virtual_hardware.mpu6050 import register_name


class I2CError(Exception):
    """Base class for bus errors."""


class I2CNack(I2CError):
    """No device acknowledged the address."""


@dataclass(frozen=True)
class I2CTransaction:
    index: int
    time: str                   # simulated HH:MM:SS.mmm
    bus: str                    # "I2C1"
    op: str                     # "READ" | "WRITE"
    address: int
    register: int
    register_name: str
    data: bytes
    ack: bool
    phases: Tuple[str, ...] = field(default_factory=tuple)

    def summary(self):
        data = " ".join(f"0x{b:02X}" for b in self.data[:6]) + (" ..." if len(self.data) > 6 else "")
        status = "" if self.ack else "  NACK"
        return (f"{self.time}  {self.bus}  {self.op} 0x{self.address:02X}  "
                f"{self.register_name} (0x{self.register:02X})  [{len(self.data)}B] {data}{status}")


class VirtualI2CBus:
    def __init__(self, clock, name="I2C1", log_size=200):
        self._clock = clock
        self.name = name
        self._devices = {}
        self.log = deque(maxlen=log_size)
        self.transaction_count = 0
        self.nack_count = 0

    # -- topology -------------------------------------------------------------
    def register_device(self, address, device):
        if not 0x08 <= address <= 0x77:
            raise ValueError(f"I2C address {address:#04x} is outside the valid 7-bit range 0x08..0x77")
        if address in self._devices:
            raise ValueError(f"I2C address {address:#04x} is already taken on {self.name}")
        self._devices[address] = device

    @property
    def devices(self):
        return sorted(self._devices)

    def reset(self):
        self.log.clear()
        self.transaction_count = 0
        self.nack_count = 0

    # -- transactions ----------------------------------------------------------
    def _log(self, op, address, register, data, ack, phases):
        self.transaction_count += 1
        tx = I2CTransaction(self.transaction_count, self._clock.timestamp(), self.name, op, address, register,
                            register_name(register), bytes(data), ack, tuple(phases))
        self.log.append(tx)
        return tx

    def _device(self, address, op, register):
        device = self._devices.get(address)
        if device is None:
            self.nack_count += 1
            self._log(op, address, register, b"", False,
                      ["START", f"ADDR 0x{address:02X} W", "NACK", "STOP"])
            raise I2CNack(f"no device at 0x{address:02X} on {self.name}")
        return device

    def read(self, address, register, length=1) -> bytes:
        device = self._device(address, "READ", register)
        data = device.read_registers(register, length)
        shown = " ".join(f"0x{b:02X}" for b in data[:3]) + (" ..." if length > 3 else "")
        self._log("READ", address, register, data, True,
                  ["START", f"ADDR 0x{address:02X} W", f"REG 0x{register:02X}", "RESTART",
                   f"ADDR 0x{address:02X} R", f"DATA {shown}", "NACK", "STOP"])
        return data

    def write(self, address, register, value):
        device = self._device(address, "WRITE", register)
        device.write_register(register, value)
        self._log("WRITE", address, register, bytes([value]), True,
                  ["START", f"ADDR 0x{address:02X} W", f"REG 0x{register:02X}", f"DATA 0x{value:02X}", "STOP"])

    def recent(self, n=8):
        return list(self.log)[-n:]

    def last(self) -> Optional[I2CTransaction]:
        return self.log[-1] if self.log else None
