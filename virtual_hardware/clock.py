"""Deterministic simulation clock: one tick == 100 ms of simulated time (10 Hz)."""

from datetime import datetime, timedelta

import numpy as np

TICK_HZ = 10
TICK_MS = 1000 // TICK_HZ
# Fixed, arbitrary epoch so simulated timestamps are reproducible (never wall-clock).
EPOCH = datetime(2026, 1, 1, 10, 24, 30)


class VirtualClock:
    """Counts ticks; every random stream is derived from the seed, so a seed fully determines a run."""

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.tick = 0

    def advance(self):
        self.tick += 1
        return self.tick

    def reset(self):
        self.tick = 0

    @property
    def time_ms(self):
        return self.tick * TICK_MS

    @property
    def time_s(self):
        return self.time_ms / 1000.0

    def timestamp(self):
        """HH:MM:SS.mmm of simulated time, as shown in the I2C / UART logs."""
        t = EPOCH + timedelta(milliseconds=self.time_ms)
        return t.strftime("%H:%M:%S.") + f"{t.microsecond // 1000:03d}"

    def rng(self, stream):
        """An independent, reproducible generator for a named stream ('vehicle', 'mpu6050', ...)."""
        key = [ord(c) for c in stream]
        return np.random.default_rng([self.seed, *key])
