"""
Virtual UART (one direction: MCU TX -> receiver).

Keeps a TX buffer the receiving side drains with read_line(), a packet counter, the last packet
and a bounded packet history. The wire time of each packet at the configured baud rate (10 bits per
byte: start + 8 data + stop) is recorded, which shows e.g. that a ~70-byte telemetry line takes ~6 ms
at 115200 baud -- well inside the 100 ms sampling period.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

DEFAULT_BAUD = 115200


@dataclass(frozen=True)
class UARTPacket:
    index: int
    time: str
    text: str
    n_bytes: int
    wire_time_ms: float


class VirtualUART:
    def __init__(self, clock, name="USART2", baud=DEFAULT_BAUD, history_size=100):
        if baud <= 0:
            raise ValueError("baud must be positive")
        self._clock = clock
        self.name = name
        self.baud = baud
        self._tx = deque()                         # bytes the receiver has not read yet
        self.history = deque(maxlen=history_size)
        self.packet_count = 0
        self.last_packet: Optional[UARTPacket] = None
        self.byte_count = 0

    def transmit(self, text):
        data = text.encode("ascii")
        self._tx.append(data)
        self.packet_count += 1
        self.byte_count += len(data)
        pkt = UARTPacket(self.packet_count, self._clock.timestamp(), text.rstrip("\r\n"), len(data),
                         len(data) * 10 * 1000.0 / self.baud)
        self.last_packet = pkt
        self.history.append(pkt)
        return pkt

    @property
    def tx_buffer_bytes(self):
        return sum(len(p) for p in self._tx)

    def read_line(self) -> Optional[bytes]:
        """Receiver side: next pending line (bytes, incl. terminator) or None."""
        return self._tx.popleft() if self._tx else None

    def reset(self):
        self._tx.clear()
        self.history.clear()
        self.packet_count = 0
        self.byte_count = 0
        self.last_packet = None

    def recent(self, n=8):
        return list(self.history)[-n:]
