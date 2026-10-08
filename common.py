"""
Shared configuration for Phase 2 (ML classification) of the
AI-Based Smart Accident Detection & Emergency Response System.

IMPORTANT: Every model trained using this configuration is trained on
SIMULATED / SYNTHETIC sensor data only (see Phase 1, sensor_simulator.py).
It is NOT validated against real-world accident data.
"""

import glob

# Sensor columns used as ML input features.
# NOTE: "scenario" (the label) and "timestamp" are intentionally excluded --
# they are not physical sensor measurements.
FEATURE_COLUMNS = [
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "vibration_level",
    "speed_kmh",
]

LABEL_COLUMN = "scenario"

# Picks up every simulated_*.csv produced by sensor_simulator.py (Phase 1).
CSV_PATTERN = "simulated_*.csv"

MODEL_PATH = "accident_classifier_model.joblib"


def find_csv_files():
    return sorted(glob.glob(CSV_PATTERN))
