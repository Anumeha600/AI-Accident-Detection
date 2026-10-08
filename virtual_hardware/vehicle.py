"""
Deterministic virtual vehicle (tick-based, 100 ms per step).

The Phase 4 generators in dataset_generator.py build whole 50-sample arrays at once, so they
cannot drive a live tick-by-tick simulation. This module is an ADAPTER: it re-expresses the same
scenario shapes (Gaussian impact envelopes, sigmoid speed drops, the same amplitude / width /
speed ranges) as noise-free PHYSICAL quantities evaluated one tick at a time. Sensor noise,
range limits and quantisation are added later by the virtual sensors, not here. The Phase 7
dataset generation is not touched.

Two scenarios do not exist in the Phase 4 dataset and are therefore new here:
Rollover and Multi-Impact Collision. The 6-class model has no label for them; whatever it
predicts for them is shown as-is.

Body frame: x forward, y left, z up. Accelerations are specific force in g (gravity included, so
a level vehicle reads az = +1 g), angular rates are in deg/s.
"""

import math
from dataclasses import dataclass, asdict

from realtime_pipeline import GPS_ORIGIN
from virtual_hardware.clock import TICK_HZ

DT = 1.0 / TICK_HZ

VIRTUAL_SCENARIOS = [
    "Normal Driving", "Pothole", "Hard Braking", "Sharp Turn",
    "Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision",
]

_M_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class VehicleState:
    tick: int
    timestamp_ms: int
    scenario: str
    vehicle_state: str          # DRIVING / POTHOLE IMPACT / HARD BRAKING / TURNING / COLLISION / ROLLOVER / STOPPED
    event_kind: str             # none / pothole / braking / turning / impact / rollover
    shock: float                # 0..~1.3 impact envelope; drives the vibration sensor
    speed_kmh: float
    accel_g: tuple              # (x, y, z) true specific force, g
    gyro_dps: tuple             # (x, y, z) true angular rate, deg/s
    acceleration_long_g: float  # forward acceleration (negative = braking)
    heading_deg: float
    roll_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    latitude: float
    longitude: float
    crashed: bool

    def as_dict(self):
        return asdict(self)


def _bump(k, center, width, amplitude):
    return amplitude * math.exp(-((k - center) ** 2) / (2 * width ** 2))


def _sigmoid(k, center, steepness, start, end):
    return start + (end - start) / (1 + math.exp(-steepness * (k - center)))


def _smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class VirtualVehicle:
    def __init__(self, clock, origin=GPS_ORIGIN):
        self._clock = clock
        self._origin = origin
        self.reset()

    # -- control -----------------------------------------------------------
    def reset(self):
        self._rng = self._clock.rng("vehicle")
        self._k = 0
        self._scenario = "Normal Driving"
        self._lat, self._lon = self._origin
        self._heading = 90.0
        self._params = {}
        self._crashed = False
        self._state = None
        self._draw_params()

    @property
    def scenario(self):
        return self._scenario

    @property
    def state(self):
        return self._state

    def set_scenario(self, scenario):
        """Start a scenario NOW (event timing is drawn relative to this moment). Position/heading persist."""
        if scenario not in VIRTUAL_SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {VIRTUAL_SCENARIOS}")
        self._scenario = scenario
        self._k = 0
        self._crashed = False
        self._draw_params()

    def _draw_params(self):
        r, s = self._rng, self._scenario
        p = {"center": int(r.integers(15, 36))}
        if s in ("Normal Driving", "Pothole"):
            p["speed0"] = r.uniform(40, 60)        # same baseline ranges as dataset_generator
        if s == "Pothole":
            p.update(width=r.uniform(0.8, 1.8), amp=r.uniform(1.0, 2.0))
        elif s == "Hard Braking":
            p.update(width=r.uniform(4, 8), amp=r.uniform(0.45, 0.85),
                     speed0=r.uniform(60, 85), speed1=r.uniform(10, 30))
        elif s == "Sharp Turn":
            p.update(width=r.uniform(3, 5.5), amp=r.uniform(0.8, 1.3), speed0=r.uniform(35, 55))
        elif s == "Minor Accident":
            p.update(width=r.uniform(1.6, 2.6), amp=r.uniform(0.8, 1.3),
                     speed0=r.uniform(35, 55), speed1=r.uniform(5, 18))
        elif s == "Severe Accident":
            p.update(width=r.uniform(1.6, 2.6), amp=r.uniform(0.85, 1.3),
                     speed0=r.uniform(45, 70), speed1=r.uniform(0, 8))
        elif s == "Rollover":
            p.update(amp=r.uniform(0.9, 1.2), speed0=r.uniform(50, 80), duration=int(r.integers(16, 22)))
        elif s == "Multi-Impact Collision":
            p.update(width=r.uniform(1.5, 2.2), amp=r.uniform(0.85, 1.2), speed0=r.uniform(55, 85),
                     gaps=(int(r.integers(5, 9)), int(r.integers(5, 9))))
        self._params = p

    # -- physics ------------------------------------------------------------
    def _profile(self, k):
        """Noise-free physical quantities at scenario tick k."""
        p, s = self._params, self._scenario
        c = p["center"]
        out = dict(ax=0.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0, speed=50.0, shock=0.0,
                   kind="none", roll=None)
        if s == "Normal Driving":
            out["speed"] = p["speed0"] + 0.5 * math.sin(k * 0.25)
        elif s == "Pothole":
            dip = _bump(k, c, p["width"], p["amp"])
            out.update(az=1.0 - dip, gx=dip * 5, speed=p["speed0"] - dip * 3, shock=dip, kind="pothole")
        elif s == "Hard Braking":
            brake = _bump(k, c, p["width"], p["amp"])
            out.update(ax=-brake, speed=_sigmoid(k, c, 0.8, p["speed0"], p["speed1"]),
                       shock=brake, kind="braking")
        elif s == "Sharp Turn":
            turn = _bump(k, c, p["width"], p["amp"])
            out.update(ay=turn * 0.5, gz=turn * 100, speed=p["speed0"] - turn * 5, shock=turn, kind="turning")
        elif s == "Minor Accident":
            imp = _bump(k, c, p["width"], p["amp"])
            out.update(ax=-imp * 2.5, ay=imp * 1.5, az=1.0 + imp * 2.0,
                       gx=imp * 40, gy=imp * 30, gz=imp * 40,
                       speed=_sigmoid(k, c, 1.0, p["speed0"], p["speed1"]), shock=imp, kind="impact")
        elif s == "Severe Accident":
            imp = _bump(k, c, p["width"], p["amp"])
            shake = math.sin(k * 5.1) * imp          # deterministic stand-in for the generator's "chaos"
            out.update(ax=-imp * 9 + shake, ay=imp * 6 + shake, az=1.0 + imp * 8 + shake,
                       gx=imp * 150, gy=imp * 120, gz=imp * 150,
                       speed=_sigmoid(k, c, 1.2, p["speed0"], p["speed1"]), shock=imp, kind="impact")
        elif s == "Rollover":
            self._rollover(out, k)
        elif s == "Multi-Impact Collision":
            self._multi_impact(out, k)
        out["speed"] = max(out["speed"], 0.0)
        return out

    def _rollover(self, out, k):
        p = self._params
        c, dur, amp = p["center"], p["duration"], p["amp"]
        x = (k - c) / dur
        roll = 180.0 * _smoothstep(x)
        rate = 180.0 * 6 * x * (1 - x) / (dur * DT) if 0 < x < 1 else 0.0   # d(roll)/dt, deg/s
        r = math.radians(roll)
        onset = _bump(k, c, 1.5, amp)
        tumble = _bump(k, c + dur * 0.35, 1.6, 0.6 * amp) + _bump(k, c + dur * 0.7, 1.6, 0.45 * amp)
        out.update(
            ax=-onset * 4.0,
            ay=math.sin(r) + onset * 3.0 + tumble * math.cos(k * 2.3) * 3.0,
            az=math.cos(r) + onset * 3.0 + tumble * 2.5,
            gx=rate, gy=_bump(k, c + dur * 0.5, 3.0, 90.0), gz=_bump(k, c + dur * 0.5, 4.0, 60.0),
            speed=_sigmoid(k, c + 6, 0.45, p["speed0"], 0.0),
            shock=max(onset, tumble), kind="rollover", roll=roll)

    def _multi_impact(self, out, k):
        p = self._params
        c1 = p["center"]
        c2, c3 = c1 + p["gaps"][0], c1 + p["gaps"][0] + p["gaps"][1]
        w = p["width"]
        hits = [_bump(k, c1, w, p["amp"]), _bump(k, c2, w, 0.65 * p["amp"]), _bump(k, c3, w, 0.4 * p["amp"])]
        imp = sum(hits)
        sgn = math.sin(k * 4.3)
        s0 = p["speed0"]
        speed = (_sigmoid(k, c1, 1.2, s0, s0 * 0.45) if k < c2 else
                 _sigmoid(k, c2, 1.2, s0 * 0.45, s0 * 0.15) if k < c3 else
                 _sigmoid(k, c3, 1.2, s0 * 0.15, 0.0))
        out.update(ax=-hits[0] * 9 - hits[1] * 5 + hits[2] * 3 * sgn, ay=imp * 4.5 * sgn + hits[0] * 2.0,
                   az=1.0 + imp * 7 + imp * sgn,
                   gx=imp * 130, gy=imp * 100 * sgn, gz=imp * 140,
                   speed=speed, shock=max(hits), kind="impact")

    def step(self):
        """Compute the vehicle state for the clock's current tick and return it."""
        k = self._k
        f = self._profile(k)
        speed = f["speed"]

        self._heading = (self._heading + f["gz"] * DT) % 360.0
        v = speed / 3.6
        h = math.radians(self._heading)
        self._lat += v * math.cos(h) * DT / _M_PER_DEG_LAT
        self._lon += v * math.sin(h) * DT / (_M_PER_DEG_LAT * math.cos(math.radians(self._lat)))

        roll = f["roll"] if f["roll"] is not None else max(-60.0, min(60.0, 5.0 * f["ay"] + 0.15 * f["gx"]))
        pitch = max(-60.0, min(60.0, -3.0 * f["ax"] + 0.1 * f["gy"]))

        if f["kind"] in ("impact", "rollover") and f["shock"] > 0.3:
            self._crashed = True
        self._state = VehicleState(
            tick=self._clock.tick, timestamp_ms=self._clock.time_ms, scenario=self._scenario,
            vehicle_state=self._label(f), event_kind=f["kind"], shock=f["shock"], speed_kmh=speed,
            accel_g=(f["ax"], f["ay"], f["az"]), gyro_dps=(f["gx"], f["gy"], f["gz"]),
            acceleration_long_g=f["ax"], heading_deg=self._heading, roll_deg=roll, pitch_deg=pitch,
            yaw_rate_dps=f["gz"], latitude=self._lat, longitude=self._lon, crashed=self._crashed)
        self._k += 1
        return self._state

    def _label(self, f):
        kind, shock = f["kind"], f["shock"]
        if self._crashed and shock <= 0.3:
            return "STOPPED (CRASHED)" if f["speed"] < 10.0 else "POST-IMPACT"
        if kind == "rollover" and (f["gx"] > 1.0 or shock > 0.3):
            return "ROLLOVER"
        if kind == "impact" and shock > 0.3:
            return "COLLISION"
        if kind == "pothole" and shock > 0.3:
            return "POTHOLE IMPACT"
        if kind == "braking" and shock > 0.2:
            return "HARD BRAKING"
        if kind == "turning" and shock > 0.2:
            return "TURNING"
        return "DRIVING"
