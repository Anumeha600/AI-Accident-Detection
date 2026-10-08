"""window_features.extract_window_features(), make_windows(): known-input calculations."""

import numpy as np
import pandas as pd
import pytest

from common import FEATURE_COLUMNS
from window_features import (
    WINDOW_SIZE, STRIDE, WINDOW_FEATURE_COLUMNS, extract_window_features, make_windows,
)


def constant_window(**overrides):
    base = dict(accel_x_g=0.0, accel_y_g=0.0, accel_z_g=1.0, gyro_x_dps=0.0, gyro_y_dps=0.0,
                gyro_z_dps=0.0, vibration_level=0.3, speed_kmh=50.0)
    base.update(overrides)
    return pd.DataFrame({c: [base[c]] * WINDOW_SIZE for c in FEATURE_COLUMNS})


def test_constants():
    assert WINDOW_SIZE == 10 and STRIDE == 10
    assert len(WINDOW_FEATURE_COLUMNS) == 4 * len(FEATURE_COLUMNS) + 4
    assert len(set(WINDOW_FEATURE_COLUMNS)) == len(WINDOW_FEATURE_COLUMNS)


def test_returns_exactly_the_declared_feature_columns():
    assert set(extract_window_features(constant_window())) == set(WINDOW_FEATURE_COLUMNS)


def test_constant_window_known_values():
    f = extract_window_features(constant_window())
    assert f["speed_kmh_mean"] == 50.0
    assert f["speed_kmh_std"] == 0.0
    assert f["speed_kmh_range"] == 0.0
    assert f["speed_kmh_first_to_last"] == 0.0
    assert f["accel_z_g_mean"] == 1.0
    assert f["accel_mag_mean"] == pytest.approx(1.0)
    assert f["accel_mag_max"] == pytest.approx(1.0)
    assert f["gyro_mag_mean"] == 0.0
    assert f["gyro_mag_max"] == 0.0


def test_linear_ramp_known_values():
    df = constant_window()
    df["speed_kmh"] = np.arange(10, dtype=float)  # 0..9
    f = extract_window_features(df)
    assert f["speed_kmh_mean"] == pytest.approx(4.5)
    assert f["speed_kmh_std"] == pytest.approx(np.sqrt(8.25))   # population std (ddof=0)
    assert f["speed_kmh_range"] == 9.0
    assert f["speed_kmh_first_to_last"] == 9.0


def test_first_to_last_is_signed_and_uses_endpoints_only():
    df = constant_window()
    df["speed_kmh"] = [60, 99, 0, 99, 0, 99, 0, 99, 0, 40]
    f = extract_window_features(df)
    assert f["speed_kmh_first_to_last"] == -20.0      # 40 - 60
    assert f["speed_kmh_range"] == 99.0


def test_magnitude_features_3_4_5():
    df = constant_window(accel_x_g=3.0, accel_y_g=4.0, accel_z_g=0.0,
                         gyro_x_dps=0.0, gyro_y_dps=6.0, gyro_z_dps=8.0)
    f = extract_window_features(df)
    assert f["accel_mag_mean"] == pytest.approx(5.0)
    assert f["accel_mag_max"] == pytest.approx(5.0)
    assert f["gyro_mag_mean"] == pytest.approx(10.0)
    assert f["gyro_mag_max"] == pytest.approx(10.0)


def test_magnitude_max_picks_the_spike():
    df = constant_window()
    df.loc[4, ["accel_x_g", "accel_y_g", "accel_z_g"]] = [0.0, 0.0, 8.0]
    f = extract_window_features(df)
    assert f["accel_mag_max"] == pytest.approx(8.0)
    assert f["accel_mag_mean"] == pytest.approx((9 * 1.0 + 8.0) / 10)


def test_input_dataframe_is_not_mutated():
    df = constant_window()
    before = df.copy()
    extract_window_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_works_on_a_slice_with_non_zero_start_index():
    big = pd.concat([constant_window(speed_kmh=10.0), constant_window(speed_kmh=70.0)],
                    ignore_index=True)
    assert extract_window_features(big.iloc[10:20])["speed_kmh_mean"] == 70.0


def test_make_windows_non_overlapping_and_drops_remainder():
    wins = list(make_windows(pd.DataFrame({"i": range(25)}), window_size=10, stride=10))
    assert [int(w["i"].iloc[0]) for w in wins] == [0, 10]     # trailing 5 rows dropped
    assert all(len(w) == 10 for w in wins)


def test_make_windows_on_50_row_recording_gives_5_windows():
    assert len(list(make_windows(pd.DataFrame({"i": range(50)})))) == 5


def test_make_windows_shorter_than_window_yields_nothing():
    assert list(make_windows(pd.DataFrame({"i": range(9)}))) == []
