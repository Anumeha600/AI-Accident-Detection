"""
Virtual vibration sensor (analog output sampled by an STM32 ADC channel).

Output unit is the pipeline's `vibration_level` (same scale as dataset_generator: road noise
|N(0, 0.3)| + 0.2 baseline, plus an event-dependent term proportional to the vehicle's impact
envelope). The per-event gain mirrors the Phase 4 generators:
pothole x3, braking x2, turning x1.5, minor/severe/multi-impact accident x5..x9.
"""

# event_kind -> gain applied to the vehicle's shock envelope
EVENT_GAIN = {"none": 0.0, "pothole": 3.0, "braking": 2.0, "turning": 1.5, "rollover": 8.0}
IMPACT_GAIN_MINOR = 5.0
IMPACT_GAIN_SEVERE = 9.0

BASELINE = 0.2
ROAD_NOISE_STD = 0.3


class VibrationSensor:
    def __init__(self, rng):
        self._rng = rng
        self.level = BASELINE

    def gain_for(self, vehicle_state):
        if vehicle_state.event_kind == "impact":
            minor = vehicle_state.scenario == "Minor Accident"
            return IMPACT_GAIN_MINOR if minor else IMPACT_GAIN_SEVERE
        return EVENT_GAIN.get(vehicle_state.event_kind, 0.0)

    def sense(self, vehicle_state):
        """Update and return the vibration level for this tick (>= 0)."""
        noise = abs(self._rng.normal(0.0, ROAD_NOISE_STD))
        self.level = max(0.0, BASELINE + noise + self.gain_for(vehicle_state) * vehicle_state.shock)
        return self.level
