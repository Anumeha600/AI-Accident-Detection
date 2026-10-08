"""Phase 7: dataset_generator_v2 -- new scenarios, ranges, IDs, reproducibility."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataset_generator as dg
import dataset_generator_v2 as g2
from common import FEATURE_COLUMNS

NEW = ["Rollover", "Multi-Impact Collision"]
CSV_COLUMNS = ["recording_id", "sample_index", "timestamp", "accel_x_g", "accel_y_g", "accel_z_g",
               "gyro_x_dps", "gyro_y_dps", "gyro_z_dps", "vibration_level", "speed_kmh", "scenario"]


@pytest.fixture(scope="module")
def small():
    """3 recordings per scenario, all 8 scenarios, fixed seed."""
    return g2.generate_dataset(recordings_per_scenario=3, seed=7)


def many(scenario, count=40, seed=0):
    rng = np.random.default_rng(seed)
    return [g2.generate_recording(rng, scenario, i) for i in range(count)]


def peak_count(rec, threshold=1.5, half_width=2):
    """Distinct local maxima of |accel - gravity vector| (min separation ~ half_width samples)."""
    a = rec[["accel_x_g", "accel_y_g", "accel_z_g"]].to_numpy()
    dev = np.linalg.norm(a - np.array([0.0, 0.0, 1.0]), axis=1)
    return sum(
        dev[i] > threshold and dev[i] == dev[max(0, i - half_width): i + half_width + 1].max()
        for i in range(len(dev))
    )


# --- structure ---------------------------------------------------------------

def test_scenario_list_is_the_six_originals_plus_two_new():
    assert g2.SCENARIOS_V2[:6] == ["Normal Driving", "Pothole", "Hard Braking", "Sharp Turn",
                                   "Minor Accident", "Severe Accident"]
    assert g2.SCENARIOS_V2[6:] == NEW and len(set(g2.SCENARIOS_V2)) == 8


def test_columns_match_phase4_csv_layout(small, project_root):
    assert list(small.columns) == CSV_COLUMNS
    assert list(pd.read_csv(project_root / dg.OUTPUT_CSV, nrows=1).columns) == CSV_COLUMNS


def test_counts_ids_labels_and_samples(small):
    assert small["recording_id"].nunique() == 24
    assert sorted(small["recording_id"].unique()) == list(range(24))
    assert set(small["scenario"]) == set(g2.SCENARIOS_V2)
    assert (small.groupby("scenario")["recording_id"].nunique() == 3).all()
    assert (small.groupby("recording_id").size() == 50).all()
    assert (small.groupby("recording_id")["scenario"].nunique() == 1).all()
    assert len(small) == 24 * 50
    # each recording_id's rows are ordered 0..49
    assert (small.groupby("recording_id")["sample_index"].apply(lambda s: list(s) == list(range(50)))).all()


def test_default_dataset_size_constants():
    assert g2.RECORDINGS_PER_SCENARIO == 100
    assert g2.RECORDINGS_PER_SCENARIO * len(g2.SCENARIOS_V2) == 800
    assert g2.OUTPUT_CSV_V2 == "expanded_synthetic_dataset_v2.csv"
    assert g2.OUTPUT_CSV_V2 != dg.OUTPUT_CSV


def test_saved_csv_has_800_recordings_of_50_samples(project_root):
    df = pd.read_csv(project_root / g2.OUTPUT_CSV_V2)
    assert df["recording_id"].nunique() == 800 and len(df) == 40000
    assert (df.groupby("scenario")["recording_id"].nunique() == 100).all()
    assert set(df["scenario"]) == set(g2.SCENARIOS_V2)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        g2.generate_recording(np.random.default_rng(0), "Meteor Strike", 0)


# --- value ranges -----------------------------------------------------------------

@pytest.mark.parametrize("scenario", g2.SCENARIOS_V2)
def test_sensor_values_are_finite_and_in_valid_ranges(scenario):
    for rec in many(scenario, count=30):
        v = rec[FEATURE_COLUMNS]
        assert np.isfinite(v.to_numpy()).all()
        assert (rec["vibration_level"] >= 0).all() and (rec["speed_kmh"] >= 0).all()
        assert rec["speed_kmh"].max() < 150
        assert rec[["accel_x_g", "accel_y_g", "accel_z_g"]].abs().to_numpy().max() <= 20.0
        assert rec[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].abs().to_numpy().max() <= 3000.0


# --- reproducibility / independence ---------------------------------------------

def test_same_seed_gives_identical_dataset():
    a = g2.generate_dataset(recordings_per_scenario=2, seed=123)
    b = g2.generate_dataset(recordings_per_scenario=2, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_dataset():
    a = g2.generate_dataset(recordings_per_scenario=2, seed=1)
    b = g2.generate_dataset(recordings_per_scenario=2, seed=2)
    assert not a[FEATURE_COLUMNS].equals(b[FEATURE_COLUMNS])


@pytest.mark.parametrize("scenario", g2.SCENARIOS_V2)
def test_recordings_within_a_scenario_are_not_identical(scenario):
    recs = many(scenario, count=6)
    arrays = [r[FEATURE_COLUMNS].to_numpy() for r in recs]
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            assert not np.array_equal(arrays[i], arrays[j])


def test_committed_csv_is_reproducible_from_default_seed(project_root):
    saved = pd.read_csv(project_root / g2.OUTPUT_CSV_V2)
    fresh = g2.generate_dataset()
    pd.testing.assert_frame_equal(saved[FEATURE_COLUMNS], fresh[FEATURE_COLUMNS], atol=1e-9, rtol=0)
    assert (saved["recording_id"].to_numpy() == fresh["recording_id"].to_numpy()).all()
    assert (saved["scenario"].to_numpy() == fresh["scenario"].to_numpy()).all()


def test_legacy_scenarios_use_the_unchanged_phase4_generators():
    # Same RNG state -> same numbers as dataset_generator.generate_recording.
    for scenario in dg.SCENARIOS:
        a = g2.generate_recording(np.random.default_rng(5), scenario, 3)
        b = dg.generate_recording(np.random.default_rng(5), scenario, 3)
        # timestamps are spaced differently on purpose (v2 honours the sample rate); data must match
        pd.testing.assert_frame_equal(a.drop(columns="timestamp"), b.drop(columns="timestamp"))


# --- sampling rate / length are configurable ---------------------------------------

@pytest.mark.parametrize("scenario", NEW)
def test_length_and_rate_configurable(scenario):
    rec = g2.generate_recording(np.random.default_rng(0), scenario, 0, n=100, sample_rate_hz=20)
    assert len(rec) == 100
    step = pd.to_datetime(rec["timestamp"]).diff().dropna().dt.total_seconds()
    assert np.allclose(step, 1 / 20)


# --- the new scenarios have the intended signatures --------------------------------

def test_rollover_has_sustained_roll_rate_and_changing_orientation():
    for rec in many("Rollover"):
        gx = rec["gyro_x_dps"].abs()
        assert gx.max() > 150                         # strong roll rate
        assert (gx > 60).sum() >= 4                   # sustained over several samples, not one spike
        # gravity direction moves: accel_z (normally ~1 g) leaves its baseline by a lot
        assert (rec["accel_z_g"] - 1).abs().max() > 0.5 or rec["accel_y_g"].abs().max() > 0.8
        assert rec["speed_kmh"].iloc[-5:].mean() < rec["speed_kmh"].iloc[:5].mean()


def test_rollover_orientation_follows_integrated_roll_rate():
    # accel (y, z) should lie on ~unit circle away from impacts: check late, quiet samples
    for rec in many("Rollover", count=20):
        tail = rec.iloc[-6:]
        mag = np.sqrt(tail.accel_y_g ** 2 + tail.accel_z_g ** 2)
        assert 0.5 < mag.mean() < 1.5


def test_multi_impact_has_multiple_distinct_peaks():
    counts = [peak_count(r) for r in many("Multi-Impact Collision")]
    assert min(counts) >= 2
    assert max(counts) >= 3


def test_single_event_scenarios_do_not_look_like_multi_impact():
    for scenario in ("Severe Accident", "Minor Accident"):
        counts = [peak_count(r) for r in many(scenario)]
        assert np.mean([c >= 2 for c in counts]) < 0.25, scenario


def test_multi_impact_timing_and_intensity_vary_between_recordings():
    recs = many("Multi-Impact Collision", count=30)
    first_peak = [int(np.argmax(np.linalg.norm(r[["accel_x_g", "accel_y_g", "accel_z_g"]] - [0, 0, 1], axis=1)))
                  for r in recs]
    peaks = [np.linalg.norm(r[["accel_x_g", "accel_y_g", "accel_z_g"]] - [0, 0, 1], axis=1).max() for r in recs]
    assert len(set(first_peak)) > 5
    assert np.std(peaks) > 0.3


def test_new_scenarios_not_separable_by_one_threshold_on_accel_magnitude():
    # Peak accel magnitude of the new scenarios overlaps the existing event classes.
    def peak(rec):
        return np.linalg.norm(rec[["accel_x_g", "accel_y_g", "accel_z_g"]].to_numpy(), axis=1).max()
    existing = [peak(r) for s in ("Minor Accident", "Severe Accident") for r in many(s, 30)]
    for s in NEW:
        mine = [peak(r) for r in many(s, 30)]
        assert min(mine) < max(existing) and max(mine) > min(existing)
