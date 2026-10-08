"""
PHASE 3 (EXPERIMENTAL) -- Windowing and feature extraction.
-------------------------------------------------------------
Splits a sensor-reading time series into short, non-overlapping windows
and computes a small set of statistical features per window. Used by
train_window_model.py and predict_window.py.

This is a SEPARATE, EXPERIMENTAL pipeline. It does not touch or replace
any Phase 2 file (common.py, train_model.py, predict.py).

WINDOW SIZE CHOICE:
    Each Phase 1 recording is 50 rows at 10 Hz (5 seconds). The sharpest
    simulated event (Pothole) has a narrow footprint of roughly 8 rows.
    A 10-sample (1-second) window is comfortably larger than that
    footprint (so a window can fully contain even the sharpest event)
    and divides the 50-row recording evenly into 5 windows.

    STRIDE = WINDOW_SIZE (non-overlapping windows) is a deliberate
    choice: it guarantees no raw row is ever reused across two windows,
    which avoids leaking near-duplicate data between the train and test
    sets. The tradeoff is fewer windows per recording.
"""

import numpy as np
import pandas as pd

from common import FEATURE_COLUMNS, LABEL_COLUMN, find_csv_files

WINDOW_SIZE = 10   # samples per window (1 second at the 10 Hz sample rate)
STRIDE = 10        # non-overlapping windows (see module docstring)

# For every raw channel we keep 4 features, each with a distinct physical
# meaning (kept deliberately small and explainable, not "every possible
# statistic"):
#   mean            - the average level during the window (e.g. an
#                      elevated vibration mean means a generally rough
#                      patch, not just one bump)
#   std             - how jittery/noisy the signal was in the window
#   range (max-min) - the peak-to-peak swing; large jolts/spikes show up
#                      here even if the mean stays near baseline
#   first_to_last   - net change from the start to the end of the window
#                      (captures a trend, e.g. speed steadily dropping
#                      during braking, or a sustained rotation)
WINDOW_FEATURE_COLUMNS = []
for _col in FEATURE_COLUMNS:
    WINDOW_FEATURE_COLUMNS += [
        f"{_col}_mean",
        f"{_col}_std",
        f"{_col}_range",
        f"{_col}_first_to_last",
    ]

# Combined accel/gyro magnitude features: an impact or spin can occur
# along any axis, so the magnitude (independent of which axis) is a
# useful summary a per-axis feature alone would miss.
WINDOW_FEATURE_COLUMNS += [
    "accel_mag_mean",
    "accel_mag_max",
    "gyro_mag_mean",
    "gyro_mag_max",
]


def make_windows(df, window_size=WINDOW_SIZE, stride=STRIDE):
    """Yield successive (non-overlapping, by default) row-windows of df."""
    n = len(df)
    start = 0
    while start + window_size <= n:
        yield df.iloc[start:start + window_size]
        start += stride


def extract_window_features(window_df):
    """Compute the WINDOW_FEATURE_COLUMNS feature dict for one window."""
    features = {}
    for col in FEATURE_COLUMNS:
        values = window_df[col].to_numpy(dtype=float)
        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std()
        features[f"{col}_range"] = values.max() - values.min()
        features[f"{col}_first_to_last"] = values[-1] - values[0]

    accel_mag = np.sqrt(
        window_df["accel_x_g"].to_numpy(dtype=float) ** 2
        + window_df["accel_y_g"].to_numpy(dtype=float) ** 2
        + window_df["accel_z_g"].to_numpy(dtype=float) ** 2
    )
    features["accel_mag_mean"] = accel_mag.mean()
    features["accel_mag_max"] = accel_mag.max()

    gyro_mag = np.sqrt(
        window_df["gyro_x_dps"].to_numpy(dtype=float) ** 2
        + window_df["gyro_y_dps"].to_numpy(dtype=float) ** 2
        + window_df["gyro_z_dps"].to_numpy(dtype=float) ** 2
    )
    features["gyro_mag_mean"] = gyro_mag.mean()
    features["gyro_mag_max"] = gyro_mag.max()

    return features


def build_windowed_dataset(csv_files=None, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Load each CSV, split it into windows, and extract features from each.
    Returns one DataFrame with one row per window: feature columns plus
    the scenario label, the source file, and the window's start index
    (kept for traceability, not used as ML features).
    """
    if csv_files is None:
        csv_files = find_csv_files()

    rows = []
    for path in csv_files:
        df = pd.read_csv(path)
        scenario = df[LABEL_COLUMN].iloc[0]
        for window_start, window_df in zip(
            range(0, len(df), stride), make_windows(df, window_size, stride)
        ):
            feats = extract_window_features(window_df)
            feats[LABEL_COLUMN] = scenario
            feats["source_file"] = path
            feats["window_start_row"] = window_start
            rows.append(feats)

    return pd.DataFrame(rows)
