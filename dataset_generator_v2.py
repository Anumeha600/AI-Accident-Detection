"""
PHASE 7 -- Expanded synthetic dataset generator, version 2 (8 scenarios).
--------------------------------------------------------------------------
Extends the Phase 4 dataset (dataset_generator.py, 6 scenarios) with two new
SIMULATED scenarios:

    Rollover               - skid/trip -> sustained roll about the
                             longitudinal axis -> ground contacts -> rest
    Multi-Impact Collision - 2-4 distinct narrow impact peaks separated by
                             recovery/disturbance periods, speed dropping
                             in steps

The six original scenarios are NOT re-implemented: their generators are
imported unchanged from dataset_generator.py, so Phase 4 and Phase 7 share
one definition of them. (Phase 7 draws FRESH random numbers for them from a
different seed, so these recordings are new samples, not copies of the
Phase 4 CSV.) Nothing here writes to, or reads from, any Phase 2-4 file.

OUTPUT
    expanded_synthetic_dataset_v2.csv   (same columns as the Phase 4 CSV)
    recording_id, sample_index, timestamp, accel_x_g, accel_y_g, accel_z_g,
    gyro_x_dps, gyro_y_dps, gyro_z_dps, vibration_level, speed_kmh, scenario

SAMPLING RATE
    Default 10 Hz x 5 s = 50 samples (what the 10-sample/1-second window in
    window_features.py assumes). `sample_rate_hz` and `n` are configurable;
    the NEW scenarios integrate physical quantities with dt = 1/sample_rate_hz
    (e.g. roll angle = integral of roll rate), while the six legacy scenarios
    are defined in sample indices (as in Phase 4) and only their timestamps
    follow the rate. Note window_features.WINDOW_SIZE stays 10 samples, so a
    different rate changes what one "window" spans in seconds.

PHYSICAL MOTIVATION OF THE NEW SCENARIOS (a simplified model, not vehicle
dynamics software). Body frame: x forward, y left, z up; roll is about x.

    Rollover
      * The accelerometer measures the specific force, which at rest is
        gravity expressed in the body frame. When the body has rolled by
        angle theta about x, that is  accel_y = sin(theta), accel_z =
        cos(theta). So a roll shows up as accel orientation CHANGING (not
        just a spike), and the vehicle can come to rest on its side
        (accel_z ~ 0, |accel_y| ~ 1) or roof (accel_z ~ -1).
      * gyro_x is the roll rate: a smooth (raised-cosine) pulse of 0.8-1.8 s
        with a 200-450 deg/s peak. theta is its running integral, so the
        total roll angle follows from the drawn pulse (roughly 80-400 deg).
      * Before the roll a lateral skid/trip: a bump in lateral acceleration
        and yaw rate (so the first phase resembles a Sharp Turn).
      * Each time the body passes a multiple of 90 deg, a ground contact
        adds a short random-direction acceleration impulse (1.5-4 g), plus
        a smaller settle impulse when the roll ends.
      * Vibration rises during the roll and contacts; speed falls toward
        0-15 km/h over the event.
    Multi-Impact Collision
      * k in {2,3,4} impacts (p = .25/.5/.25). First impact at 10-30% of
        the recording, later ones 5-11 samples apart. Each impact is a
        NARROW bump (sigma 0.8-1.5 samples) of 2.5-7 g peak along a random
        direction, with its own random gyro kick (30-130 deg/s per axis).
      * Between impacts: smaller, wider disturbance bumps (secondary
        scraping/yaw wobble, 0.5-1.2 g).
      * Speed drops in a step at each impact (random split of a total drop
        to 0-10 km/h) with partial recovery of ordinary driving noise.

ASSUMPTIONS / LIMITS
    All ranges above are hand-chosen by the author, not derived from real
    telemetry. A real MPU6050 as configured in stm32_firmware/ saturates at
    +/-2 g and +/-250 deg/s; like the Phase 4 Severe Accident scenario, the
    synthetic impacts and roll rates here exceed that range. Window labels
    are the label of the whole recording (as in Phase 4), so windows taken
    before an event starts look like Normal Driving but carry the event label.

IMPORTANT: All output is SIMULATED / SYNTHETIC data. It does not come from a
real accelerometer, gyroscope, vibration, or speed sensor, and must not be
treated as validated real-world data.
"""

from datetime import datetime, timedelta
from typing import Callable, Dict

import numpy as np
import pandas as pd

import dataset_generator as dg
from dataset_generator import (
    NUM_SAMPLES, RAW_FEATURE_COLUMNS, SAMPLE_RATE_HZ,
    _gaussian_bump, _noise, _sigmoid_transition,
)

RECORDINGS_PER_SCENARIO = 100
DATASET_SEED_V2 = 200  # fixes the WHOLE run; deliberately different from Phase 4's seed (100)

NEW_SCENARIOS = ["Rollover", "Multi-Impact Collision"]
SCENARIOS_V2 = list(dg.SCENARIOS) + NEW_SCENARIOS

OUTPUT_CSV_V2 = "expanded_synthetic_dataset_v2.csv"

RawChannels = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# NEW SCENARIO GENERATORS
# ---------------------------------------------------------------------------

def _gen_rollover(rng: np.random.Generator, n: int, dt: float) -> RawChannels:
    """Skid -> roll about the x axis (gravity rotates in the y-z plane) -> rest."""
    idx = np.arange(n)

    dur = rng.uniform(0.8, 1.8) / dt                    # roll duration, in samples
    peak_rate = rng.uniform(200.0, 450.0)               # deg/s
    lo, hi = max(2, int(0.24 * n)), max(3, int(0.5 * n))
    t0 = int(min(rng.integers(lo, hi + 1), max(2, n - 2 - dur)))   # roll onset (sample)
    sign = rng.choice([-1.0, 1.0])                      # roll direction

    # Raised-cosine roll-rate pulse (0 -> peak -> 0) over [t0, t0 + dur].
    u = np.clip((idx - t0) / dur, 0.0, 1.0)
    profile = np.where((idx > t0) & (idx < t0 + dur), 0.5 * (1.0 - np.cos(2.0 * np.pi * u)), 0.0)
    roll_rate = sign * peak_rate * profile              # deg/s about x
    theta = np.cumsum(roll_rate) * dt                   # roll angle (deg), integral of the rate
    theta_rad = np.deg2rad(theta)

    # Pre-roll skid/trip: lateral acceleration + yaw, like a (violent) sharp turn.
    skid = _gaussian_bump(n, t0 - rng.uniform(1.0, 3.0), rng.uniform(1.5, 3.0), rng.uniform(0.3, 0.7))

    # Ground contacts every time the body passes a multiple of 90 deg, + a
    # settle impulse when the roll ends. Random direction in the y-z plane.
    contact_times = list(np.flatnonzero(np.diff(np.floor(np.abs(theta) / 90.0), prepend=0.0) > 0))
    contact_times.append(min(n - 1, int(t0 + dur) + 1))
    imp_y = np.zeros(n)
    imp_z = np.zeros(n)
    imp_env = np.zeros(n)
    for k, ct in enumerate(contact_times):
        amp = rng.uniform(0.8, 2.0) if k == len(contact_times) - 1 else rng.uniform(1.5, 4.0)
        bump = _gaussian_bump(n, ct, 0.9, amp)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        imp_y += bump * np.cos(phi)
        imp_z += bump * np.sin(phi)
        imp_env += bump

    start_speed = rng.uniform(45.0, 80.0)
    end_speed = rng.uniform(0.0, 15.0)
    return {
        "accel_x": _noise(rng, n, 0.1) - 0.3 * profile + 0.2 * imp_env * rng.choice([-1.0, 1.0]),
        "accel_y": np.sin(theta_rad) + sign * skid + _noise(rng, n, 0.12) + imp_y,
        "accel_z": np.cos(theta_rad) + _noise(rng, n, 0.1) + imp_z,
        "gyro_x": roll_rate + _noise(rng, n, 4.0),
        "gyro_y": _noise(rng, n, 4.0) + rng.uniform(-0.25, 0.25) * roll_rate + 8.0 * imp_env,
        "gyro_z": _noise(rng, n, 3.0) + sign * skid * 60.0 + rng.uniform(-0.3, 0.3) * roll_rate,
        "vibration": np.abs(_noise(rng, n, 0.5)) + 0.3 + profile * rng.uniform(3.0, 6.0) + 2.0 * imp_env,
        "speed": _sigmoid_transition(n, t0 + dur / 2.0, 0.5, start_speed, end_speed) + _noise(rng, n, 1.5),
    }


def _gen_multi_impact(rng: np.random.Generator, n: int, dt: float) -> RawChannels:
    """2-4 distinct narrow impacts with disturbances between them."""
    k = int(rng.choice([2, 3, 4], p=[0.25, 0.5, 0.25]))
    centers = [int(rng.integers(max(2, int(0.1 * n)), int(0.3 * n) + 1))]
    while len(centers) < k:
        nxt = centers[-1] + int(rng.integers(5, 12))
        if nxt > n - 4:
            break
        centers.append(nxt)

    acc = np.zeros((3, n))
    gyr = np.zeros((3, n))
    vib = np.zeros(n)
    for c in centers:
        bump = _gaussian_bump(n, c, rng.uniform(0.8, 1.5), 1.0)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        acc += np.outer(direction * rng.uniform(2.5, 7.0), bump)
        gyr += np.outer(rng.uniform(30.0, 130.0, 3) * rng.choice([-1.0, 1.0], 3), bump)
        vib += bump * rng.uniform(4.0, 8.0)

    # Secondary disturbances (scrape/yaw wobble) between consecutive impacts.
    for a, b in zip(centers[:-1], centers[1:]):
        wide = _gaussian_bump(n, (a + b) / 2.0, rng.uniform(2.0, 3.0), 1.0)
        acc += np.outer(rng.normal(size=3) * rng.uniform(0.5, 1.2) / 1.7, wide)
        gyr[2] += wide * rng.uniform(-60.0, 60.0)
        vib += wide * rng.uniform(1.0, 2.0)

    # Speed falls in a step at each impact: a random split of the total drop.
    start_speed = rng.uniform(40.0, 75.0)
    end_speed = rng.uniform(0.0, 10.0)
    shares = rng.dirichlet(np.ones(len(centers)))
    speed = np.full(n, start_speed)
    for share, c in zip(shares, centers):
        speed += _sigmoid_transition(n, c, 1.2, 0.0, -share * (start_speed - end_speed))

    return {
        "accel_x": _noise(rng, n, 0.12) + acc[0],
        "accel_y": _noise(rng, n, 0.12) + acc[1],
        "accel_z": 1.0 + _noise(rng, n, 0.12) + acc[2],
        "gyro_x": _noise(rng, n, 4.0) + gyr[0],
        "gyro_y": _noise(rng, n, 4.0) + gyr[1],
        "gyro_z": _noise(rng, n, 4.0) + gyr[2],
        "vibration": np.abs(_noise(rng, n, 0.5)) + 0.3 + vib,
        "speed": speed + _noise(rng, n, 1.2),
    }


_NEW_GENERATORS: Dict[str, Callable[[np.random.Generator, int, float], RawChannels]] = {
    "Rollover": _gen_rollover,
    "Multi-Impact Collision": _gen_multi_impact,
}


# ---------------------------------------------------------------------------
# RECORDING / DATASET ASSEMBLY
# ---------------------------------------------------------------------------

def generate_recording(
    rng: np.random.Generator,
    scenario: str,
    recording_id: int,
    n: int = NUM_SAMPLES,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """Generate one synthetic recording (n rows) for any of the 8 scenarios."""
    if scenario in _NEW_GENERATORS:
        raw = _NEW_GENERATORS[scenario](rng, n, 1.0 / sample_rate_hz)
    elif scenario in dg.GENERATORS:
        raw = dg.GENERATORS[scenario](rng, n)       # unchanged Phase 4 generators
    else:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS_V2}")

    # Timestamps only for readability/ordering; recordings are spaced apart
    # in time. Not used as an ML feature downstream.
    start_time = datetime(2026, 1, 1) + timedelta(seconds=int(recording_id) * (n / sample_rate_hz + 5))
    timestamps = [start_time + timedelta(seconds=i / sample_rate_hz) for i in range(n)]

    df = pd.DataFrame({
        "recording_id": recording_id,
        "sample_index": np.arange(n),
        "timestamp": timestamps,
        "accel_x_g": raw["accel_x"],
        "accel_y_g": raw["accel_y"],
        "accel_z_g": raw["accel_z"],
        "gyro_x_dps": raw["gyro_x"],
        "gyro_y_dps": raw["gyro_y"],
        "gyro_z_dps": raw["gyro_z"],
        "vibration_level": np.clip(raw["vibration"], 0.0, None),
        "speed_kmh": np.clip(raw["speed"], 0.0, None),
        "scenario": scenario,
    })
    df[RAW_FEATURE_COLUMNS] = df[RAW_FEATURE_COLUMNS].round(3)
    return df


def generate_dataset(
    recordings_per_scenario: int = RECORDINGS_PER_SCENARIO,
    seed: int = DATASET_SEED_V2,
    n: int = NUM_SAMPLES,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """Many independent recordings per scenario, each with a unique recording_id.

    One seeded Generator drives the whole run, so the same arguments always
    give the identical dataset; every recording consumes fresh draws.
    """
    rng = np.random.default_rng(seed)
    recordings = []
    recording_id = 0
    for scenario in SCENARIOS_V2:
        for _ in range(recordings_per_scenario):
            recordings.append(generate_recording(rng, scenario, recording_id, n, sample_rate_hz))
            recording_id += 1
    return pd.concat(recordings, ignore_index=True)


def main() -> None:
    print("=" * 70)
    print("PHASE 7: Generating the 8-scenario EXPANDED synthetic dataset (v2)")
    print(f"({RECORDINGS_PER_SCENARIO} independent recordings per scenario, seed {DATASET_SEED_V2})")
    print("ALL DATA IS SIMULATED/SYNTHETIC -- see the module docstring.")
    print("=" * 70 + "\n")

    dataset = generate_dataset()
    print(f"Total independent recordings: {dataset['recording_id'].nunique()}")
    print(f"Total rows: {len(dataset)}  |  columns: {len(dataset.columns)}\n")
    print("Recordings per scenario:")
    print(dataset.groupby("scenario")["recording_id"].nunique().sort_index().to_string())

    dataset.to_csv(OUTPUT_CSV_V2, index=False)
    print(f"\nSaved dataset to: {OUTPUT_CSV_V2}")
    print("(expanded_synthetic_dataset.csv, the Phase 4 dataset, was NOT touched.)")


if __name__ == "__main__":
    main()
