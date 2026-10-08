"""
Virtual GPS receiver -- SIMULATED GPS. Coordinates follow the virtual vehicle (which starts at an
arbitrary point, see realtime_pipeline.GPS_ORIGIN); nothing here is a real position.

The receiver needs fix_after_ticks ticks after power-up/reset to report a fix (default 1 = warm start; a
cold start would report speed 0 until the fix, which the 1-second window features misread as a jump), and adds
~0.3 m of deterministic jitter. A crashed, stationary vehicle therefore keeps (approximately) its
last position.
"""

FIX_AFTER_TICKS = 1          # warm start: fix on the first tick (cold start = pass a larger value)
SPEED_NOISE_KMH = 2.0       # speed-channel jitter; equals the speed noise std assumed by dataset_generator
JITTER_DEG = 3e-6          # ~0.3 m


class VirtualGPS:
    def __init__(self, rng, fix_after_ticks=FIX_AFTER_TICKS):
        self._rng = rng
        self.fix_after_ticks = fix_after_ticks
        self.reset()

    def reset(self):
        self._ticks = 0
        self.fix = False
        self.latitude = 0.0
        self.longitude = 0.0
        self.speed_kmh = 0.0
        self.heading_deg = 0.0

    def sense(self, vehicle_state):
        self._ticks += 1
        self.fix = self._ticks >= self.fix_after_ticks
        if self.fix:
            self.latitude = vehicle_state.latitude + self._rng.normal(0.0, JITTER_DEG)
            self.longitude = vehicle_state.longitude + self._rng.normal(0.0, JITTER_DEG)
            self.speed_kmh = max(0.0, vehicle_state.speed_kmh + self._rng.normal(0.0, SPEED_NOISE_KMH))
            self.heading_deg = vehicle_state.heading_deg
        return self.report()

    def report(self):
        return {"fix": self.fix, "latitude": self.latitude, "longitude": self.longitude,
                "speed_kmh": self.speed_kmh, "heading_deg": self.heading_deg}
