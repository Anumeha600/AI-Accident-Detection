"""
PHASE 3 (EXPERIMENTAL) -- Prediction interface for the window-based model.
------------------------------------------------------------------------
Loads the model saved by train_window_model.py and exposes
predict_window_event(), which takes a short window of consecutive
sensor readings and returns the predicted event plus confidence.

DESIGN NOTE (for future phases):
    predict_window_event() takes a list/DataFrame of raw sensor-reading
    rows (the same shape Phase 2's predict_event() takes one of). In
    real-time use, the caller is responsible for buffering the last
    WINDOW_SIZE readings (e.g. from the STM32 over serial) and passing
    them in here -- nothing else about this function needs to change.

IMPORTANT:
    Trained ONLY on simulated/synthetic data (see train_window_model.py).
    Not validated against real-world accident data.
"""

import joblib
import pandas as pd

from common import FEATURE_COLUMNS, find_csv_files
from window_features import (
    WINDOW_SIZE,
    STRIDE,
    WINDOW_FEATURE_COLUMNS,
    extract_window_features,
    make_windows,
)

WINDOW_MODEL_PATH = "accident_classifier_model_windowed.joblib"

_model = None


def load_model():
    """Load the trained window-based model from disk (cached after first call)."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(WINDOW_MODEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No trained window model found at '{WINDOW_MODEL_PATH}'. "
                "Run train_window_model.py first."
            )
    return _model


def predict_window_event(window_readings):
    """
    Predict the driving/accident event for one window of sensor readings.

    Parameters
    ----------
    window_readings : list[dict] or pandas.DataFrame
        Exactly WINDOW_SIZE consecutive readings, each containing the
        keys in common.FEATURE_COLUMNS (accel_x_g, accel_y_g, ...,
        speed_kmh), in time order.

    Returns
    -------
    predicted_label : str
    confidence : float
    class_probabilities : dict
        Same shape as Phase 2's predict_event(), so callers can treat
        both the same way.
    """
    model = load_model()

    window_df = (
        window_readings
        if isinstance(window_readings, pd.DataFrame)
        else pd.DataFrame(window_readings)
    )

    if len(window_df) != WINDOW_SIZE:
        raise ValueError(
            f"Expected exactly {WINDOW_SIZE} readings in a window, got {len(window_df)}."
        )
    missing = [c for c in FEATURE_COLUMNS if c not in window_df.columns]
    if missing:
        raise ValueError(f"window_readings is missing required keys: {missing}")

    features = extract_window_features(window_df)
    X = pd.DataFrame([features], columns=WINDOW_FEATURE_COLUMNS)

    predicted_label = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))
    confidence = class_probabilities[predicted_label]

    return predicted_label, confidence, class_probabilities


def _demo():
    print("=" * 70)
    print("WINDOW-BASED PREDICTION DEMO -- SIMULATED/SYNTHETIC data only")
    print("(experimental Phase 3 model; not validated on real accidents)")
    print("=" * 70 + "\n")

    csv_files = find_csv_files()
    if not csv_files:
        print("No simulated_*.csv files found -- run sensor_simulator.py first.")
        return

    for path in csv_files:
        df = pd.read_csv(path)
        true_scenario = df["scenario"].iloc[0]
        windows = list(make_windows(df, WINDOW_SIZE, STRIDE))
        event_window = windows[len(windows) // 2]  # window covering the event peak
        label, confidence, _ = predict_window_event(event_window)
        match = "OK" if label == true_scenario else "MISMATCH"
        print(
            f"[{match}] Actual: {true_scenario:16s} -> Predicted: {label:16s} "
            f"(confidence: {confidence:.1%})"
        )


if __name__ == "__main__":
    _demo()
