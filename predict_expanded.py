"""
PHASE 5 -- Prediction wrapper for the Phase 4 (expanded-dataset)
window-based model.
---------------------------------------------------------------------
Mirrors Phase 3's predict_window.py, but points at the Phase 4 model
(accident_classifier_model_expanded.joblib, trained on 600 independent
synthetic recordings with a recording-level train/test split -- see
train_expanded_model.py). Reuses the exact same feature-extraction code
Phase 3 and Phase 4 both use (window_features.py), so predictions made
here are computed identically to how the model was evaluated -- no
classifier logic is duplicated.

This module does NOT retrain or modify the Phase 4 model in any way; it
only loads the file already saved by train_expanded_model.py.

IMPORTANT: Trained ONLY on simulated/synthetic data. Not validated
against real-world accident data.
"""

import joblib
import pandas as pd

from common import FEATURE_COLUMNS
from window_features import WINDOW_SIZE, WINDOW_FEATURE_COLUMNS, extract_window_features
from train_expanded_model import EXPANDED_MODEL_PATH

_model = None


def load_model():
    """Load the Phase 4 expanded-dataset model (cached after first call)."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(EXPANDED_MODEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No trained expanded model found at '{EXPANDED_MODEL_PATH}'. "
                "Run train_expanded_model.py first."
            )
    return _model


def predict_window_event(window_readings):
    """
    Predict the driving/accident event for one window of sensor readings,
    using the Phase 4 expanded-dataset model.

    Parameters
    ----------
    window_readings : list[dict] or pandas.DataFrame
        Exactly WINDOW_SIZE consecutive readings, each containing the
        keys in common.FEATURE_COLUMNS, in time order.

    Returns
    -------
    predicted_label : str
    confidence : float
    class_probabilities : dict
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
