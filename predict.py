"""
PHASE 2 -- Prediction interface for the trained accident-event classifier.
----------------------------------------------------------------------------
Loads the model saved by train_model.py and exposes predict_event(), a
single function that takes one sensor reading and returns the predicted
event plus a confidence score.

DESIGN NOTE (for future phases):
    predict_event() is the function to call from real-time code later
    (e.g. code reading live values over serial from an STM32). Nothing
    about training or the model format needs to change for that --
    only where the sensor_reading dictionary comes from.

IMPORTANT:
    The underlying model was trained ONLY on simulated/synthetic data
    (see sensor_simulator.py and train_model.py). Predictions are
    illustrative only and are NOT validated against real-world
    accident data.
"""

import joblib
import pandas as pd

from common import FEATURE_COLUMNS, MODEL_PATH, find_csv_files

_model = None  # loaded lazily, once, on first use


def load_model():
    """Load the trained model from disk (cached after the first call)."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'. "
                "Run train_model.py first."
            )
    return _model


def predict_event(sensor_reading: dict):
    """
    Predict the driving/accident event for one sensor reading.

    Parameters
    ----------
    sensor_reading : dict
        Must contain all keys listed in common.FEATURE_COLUMNS, e.g.:
        {
            "accel_x_g": 0.02, "accel_y_g": -0.01, "accel_z_g": 1.0,
            "gyro_x_dps": 0.5, "gyro_y_dps": -0.3, "gyro_z_dps": 0.1,
            "vibration_level": 0.3, "speed_kmh": 50.0,
        }

    Returns
    -------
    predicted_label : str
        The predicted scenario name, e.g. "Severe Accident".
    confidence : float
        The model's probability for the predicted label (0-1).
    class_probabilities : dict
        Probability for every possible scenario label.
    """
    model = load_model()

    missing = [c for c in FEATURE_COLUMNS if c not in sensor_reading]
    if missing:
        raise ValueError(f"sensor_reading is missing required keys: {missing}")

    # Built as a DataFrame (not a bare array) so column names match what
    # the model was trained on -- avoids an sklearn "missing feature
    # names" warning.
    X = pd.DataFrame([sensor_reading], columns=FEATURE_COLUMNS)

    predicted_label = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))
    confidence = class_probabilities[predicted_label]

    return predicted_label, confidence, class_probabilities


def _load_demo_examples():
    """
    Pull one real (simulated) row from the middle of each available CSV
    file to use as a quick demo -- for event scenarios this lands near
    the peak of the simulated event.
    """
    examples = {}
    for path in find_csv_files():
        df = pd.read_csv(path)
        if df.empty:
            continue
        mid_row = df.iloc[len(df) // 2]
        examples[mid_row["scenario"]] = {col: mid_row[col] for col in FEATURE_COLUMNS}
    return examples


def _demo():
    print("=" * 70)
    print("PREDICTION DEMO -- using SIMULATED/SYNTHETIC sensor readings")
    print("(NOT real sensor data; model NOT validated on real accidents)")
    print("=" * 70 + "\n")

    examples = _load_demo_examples()
    if not examples:
        print("No simulated_*.csv files found -- run sensor_simulator.py first.")
        return

    for true_scenario, reading in examples.items():
        label, confidence, _ = predict_event(reading)
        match = "OK" if label == true_scenario else "MISMATCH"
        print(f"[{match}] Actual scenario: {true_scenario:16s} "
              f"-> Predicted: {label:16s} (confidence: {confidence:.1%})")


if __name__ == "__main__":
    _demo()
