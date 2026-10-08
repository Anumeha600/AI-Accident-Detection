"""
PHASE 4 -- Expanded synthetic dataset generator.
----------------------------------------------------
Generates MANY independent synthetic recordings per scenario (default
100 each -- 600 total), instead of the single fixed recording each
scenario had in Phase 1. Each recording randomizes noise, event
timing, event intensity, event duration, and baseline speed within
documented, realistic ranges, so recordings are genuinely different
samples of the same scenario -- not near-duplicates of one another.

Every row is tagged with a `recording_id`, shared by all 50 rows of
that recording. Downstream scripts (train_expanded_model.py) split
train/test on recording_id BEFORE windowing, so windows from the same
recording can never appear on both sides of the split.

ASSUMPTIONS / DOCUMENTED RANDOMIZATION RANGES
(kept close to Phase 1's original constants -- this is jitter around a
physically-motivated center, not values tuned to flatter a classifier):

    - event_center: randomized per recording within samples 15-35 of
      the 50-sample (5 s) recording, instead of always sample 25 --
      real events don't always land at the exact midpoint of a
      logging window.
    - event intensity: amplitude scaled by a random multiplier,
      different range per scenario (see each _gen_* function) --
      reflecting that no two real potholes/impacts/braking events are
      physically identical.
    - event width: duration scaled by a random multiplier -- varies
      how long the event lasts.
    - baseline speed: drawn from a scenario-specific realistic range
      instead of one fixed constant.
    - sensor noise: freshly drawn per recording from a single
      reproducible random generator (seeded once for the whole run,
      so regenerating the dataset gives the same result) -- no two
      recordings share identical noise.

IMPORTANT: All output is SIMULATED / SYNTHETIC data. It does not come
from a real accelerometer, gyroscope, vibration, or speed sensor, and
must not be treated as validated real-world data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SAMPLE_RATE_HZ = 10
DURATION_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE_HZ * DURATION_SECONDS  # 50

RECORDINGS_PER_SCENARIO = 100
DATASET_SEED = 100  # fixes the WHOLE generation run so it is reproducible

SCENARIOS = [
    "Normal Driving",
    "Pothole",
    "Hard Braking",
    "Sharp Turn",
    "Minor Accident",
    "Severe Accident",
]

OUTPUT_CSV = "expanded_synthetic_dataset.csv"

RAW_FEATURE_COLUMNS = [
    "accel_x_g", "accel_y_g", "accel_z_g",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "vibration_level", "speed_kmh",
]


# ---------------------------------------------------------------------------
# MATH HELPERS (each recording gets its own random draws via `rng`)
# ---------------------------------------------------------------------------

def _noise(rng, n, std):
    return rng.normal(0.0, std, n)


def _gaussian_bump(n, center, width, amplitude):
    idx = np.arange(n)
    return amplitude * np.exp(-((idx - center) ** 2) / (2 * width ** 2))


def _sigmoid_transition(n, center, steepness, start_value, end_value):
    idx = np.arange(n)
    s = 1 / (1 + np.exp(-steepness * (idx - center)))
    return start_value + (end_value - start_value) * s


def _random_event_center(rng, low=15, high=35):
    """Event timing varies per recording instead of always sample 25."""
    return rng.integers(low, high + 1)


# ---------------------------------------------------------------------------
# PER-SCENARIO GENERATORS
# Each draws its own randomized timing/intensity/speed, then builds the
# 8 sensor-channel arrays the same way Phase 1 did.
# ---------------------------------------------------------------------------

def _gen_normal_driving(rng, n):
    baseline_speed = rng.uniform(40, 60)
    return {
        "accel_x": _noise(rng, n, 0.05),
        "accel_y": _noise(rng, n, 0.05),
        "accel_z": 1.0 + _noise(rng, n, 0.03),
        "gyro_x": _noise(rng, n, 1.0),
        "gyro_y": _noise(rng, n, 1.0),
        "gyro_z": _noise(rng, n, 1.0),
        "vibration": np.abs(_noise(rng, n, 0.3)) + 0.2,
        "speed": baseline_speed + _noise(rng, n, 2),
    }


def _gen_pothole(rng, n):
    center = _random_event_center(rng)
    width = rng.uniform(0.8, 1.8)      # sharp, brief -- jittered duration
    amplitude = rng.uniform(1.0, 2.0)  # event intensity varies
    baseline_speed = rng.uniform(40, 60)
    dip = _gaussian_bump(n, center, width, amplitude)
    return {
        "accel_x": _noise(rng, n, 0.05),
        "accel_y": _noise(rng, n, 0.05),
        "accel_z": 1.0 + _noise(rng, n, 0.03) - dip,
        "gyro_x": _noise(rng, n, 1.0) + dip * 5,
        "gyro_y": _noise(rng, n, 1.0),
        "gyro_z": _noise(rng, n, 1.0),
        "vibration": np.abs(_noise(rng, n, 0.3)) + 0.2 + dip * 3,
        "speed": baseline_speed + _noise(rng, n, 2) - dip * 3,
    }


def _gen_hard_braking(rng, n):
    center = _random_event_center(rng)
    width = rng.uniform(4, 8)
    amplitude = rng.uniform(0.45, 0.85)
    start_speed = rng.uniform(60, 85)
    end_speed = rng.uniform(10, 30)
    brake = _gaussian_bump(n, center, width, amplitude)
    return {
        "accel_x": _noise(rng, n, 0.05) - brake,
        "accel_y": _noise(rng, n, 0.05),
        "accel_z": 1.0 + _noise(rng, n, 0.03),
        "gyro_x": _noise(rng, n, 1.0),
        "gyro_y": _noise(rng, n, 1.0),
        "gyro_z": _noise(rng, n, 1.0),
        "vibration": np.abs(_noise(rng, n, 0.3)) + 0.2 + brake * 2,
        "speed": _sigmoid_transition(n, center, 0.8, start_speed, end_speed) + _noise(rng, n, 1.5),
    }


def _gen_sharp_turn(rng, n):
    center = _random_event_center(rng)
    width = rng.uniform(3, 5.5)
    amplitude = rng.uniform(0.8, 1.3)
    baseline_speed = rng.uniform(35, 55)
    turn = _gaussian_bump(n, center, width, amplitude)
    return {
        "accel_x": _noise(rng, n, 0.05),
        "accel_y": _noise(rng, n, 0.05) + turn * 0.5,
        "accel_z": 1.0 + _noise(rng, n, 0.03),
        "gyro_x": _noise(rng, n, 1.0),
        "gyro_y": _noise(rng, n, 1.0),
        "gyro_z": _noise(rng, n, 1.0) + turn * 100,
        "vibration": np.abs(_noise(rng, n, 0.3)) + 0.2 + turn * 1.5,
        "speed": baseline_speed + _noise(rng, n, 2) - turn * 5,
    }


def _gen_minor_accident(rng, n):
    center = _random_event_center(rng)
    width = rng.uniform(1.6, 2.6)
    amplitude = rng.uniform(0.8, 1.3)
    start_speed = rng.uniform(35, 55)
    end_speed = rng.uniform(5, 18)
    impact = _gaussian_bump(n, center, width, amplitude)
    return {
        "accel_x": _noise(rng, n, 0.1) - impact * 2.5,
        "accel_y": _noise(rng, n, 0.1) + impact * 1.5,
        "accel_z": 1.0 + _noise(rng, n, 0.1) + impact * 2.0,
        "gyro_x": _noise(rng, n, 3) + impact * 40,
        "gyro_y": _noise(rng, n, 3) + impact * 30,
        "gyro_z": _noise(rng, n, 3) + impact * 40,
        "vibration": np.abs(_noise(rng, n, 0.5)) + 0.3 + impact * 5,
        "speed": _sigmoid_transition(n, center, 1.0, start_speed, end_speed) + _noise(rng, n, 1.5),
    }


def _gen_severe_accident(rng, n):
    center = _random_event_center(rng)
    width = rng.uniform(1.6, 2.6)
    amplitude = rng.uniform(0.85, 1.3)
    start_speed = rng.uniform(45, 70)
    end_speed = rng.uniform(0, 8)
    impact = _gaussian_bump(n, center, width, amplitude)
    chaos = _noise(rng, n, 1.0) * impact  # extra chaotic shaking during impact
    return {
        "accel_x": _noise(rng, n, 0.15) - impact * 9 + chaos,
        "accel_y": _noise(rng, n, 0.15) + impact * 6 + chaos,
        "accel_z": 1.0 + _noise(rng, n, 0.15) + impact * 8 + chaos,
        "gyro_x": _noise(rng, n, 5) + impact * 150,
        "gyro_y": _noise(rng, n, 5) + impact * 120,
        "gyro_z": _noise(rng, n, 5) + impact * 150,
        "vibration": np.abs(_noise(rng, n, 0.5)) + 0.3 + impact * 9,
        "speed": _sigmoid_transition(n, center, 1.2, start_speed, end_speed) + np.abs(_noise(rng, n, 1.0)),
    }


GENERATORS = {
    "Normal Driving": _gen_normal_driving,
    "Pothole": _gen_pothole,
    "Hard Braking": _gen_hard_braking,
    "Sharp Turn": _gen_sharp_turn,
    "Minor Accident": _gen_minor_accident,
    "Severe Accident": _gen_severe_accident,
}


def generate_recording(rng, scenario, recording_id, n=NUM_SAMPLES, sample_rate_hz=SAMPLE_RATE_HZ):
    """Generate one 50-row synthetic recording for the given scenario."""
    raw = GENERATORS[scenario](rng, n)

    # Timestamps are only for readability/ordering -- spaced so recordings
    # don't overlap in time; not used as an ML feature downstream.
    start_time = datetime(2026, 1, 1) + timedelta(seconds=int(recording_id) * (n + 5))
    timestamps = [start_time + timedelta(seconds=i / sample_rate_hz) for i in range(n)]

    speed = np.clip(raw["speed"], a_min=0, a_max=None)

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
        "vibration_level": np.clip(raw["vibration"], a_min=0, a_max=None),
        "speed_kmh": speed,
        "scenario": scenario,
    })
    df[RAW_FEATURE_COLUMNS] = df[RAW_FEATURE_COLUMNS].round(3)
    return df


def generate_dataset(recordings_per_scenario=RECORDINGS_PER_SCENARIO, seed=DATASET_SEED):
    """Generate the full expanded dataset: many independent recordings
    per scenario, each with its own recording_id."""
    rng = np.random.default_rng(seed)
    all_recordings = []
    recording_id = 0
    for scenario in SCENARIOS:
        for _ in range(recordings_per_scenario):
            all_recordings.append(generate_recording(rng, scenario, recording_id))
            recording_id += 1
    return pd.concat(all_recordings, ignore_index=True)


def main():
    print("=" * 70)
    print("PHASE 4: Generating EXPANDED synthetic dataset")
    print(f"({RECORDINGS_PER_SCENARIO} independent recordings per scenario)")
    print("ALL DATA IS SIMULATED/SYNTHETIC -- see module docstring for the")
    print("randomization ranges used (event timing, intensity, width,")
    print("baseline speed all vary per recording).")
    print("=" * 70 + "\n")

    dataset = generate_dataset()

    print(f"Total independent recordings: {dataset['recording_id'].nunique()}")
    print(f"Total rows (recordings x 50 samples): {len(dataset)}\n")
    print("Recordings per scenario:")
    print(dataset.groupby("scenario")["recording_id"].nunique().sort_index().to_string())

    dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved expanded dataset to: {OUTPUT_CSV}")
    print("\nNOTE: This is SIMULATED/SYNTHETIC data only -- see this file's")
    print("module docstring for exactly what was randomized and why.")


if __name__ == "__main__":
    main()
