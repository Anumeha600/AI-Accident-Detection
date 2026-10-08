"""
PHASE 7 -- Prediction wrapper for the 8-scenario window model (v2).
---------------------------------------------------------------------
Same interface as predict_expanded.py (Phase 5's wrapper for the Phase 4
model), but loads ONLY accident_classifier_model_expanded_v2.joblib. It
reuses window_features.extract_window_features, so predictions are computed
exactly as the model was trained/evaluated. It never retrains or modifies
the model.

IMPORTANT: Trained ONLY on simulated/synthetic data. Not validated against
real-world accident data.
"""

from typing import Dict, List, Mapping, Sequence, Tuple, Union

import joblib
import pandas as pd

from common import FEATURE_COLUMNS
from window_features import WINDOW_SIZE, WINDOW_FEATURE_COLUMNS, extract_window_features
from train_expanded_model_v2 import EXPANDED_V2_MODEL_PATH

_model = None


def load_model():
    """Load the Phase 7 (8-scenario) model (cached after first call)."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(EXPANDED_V2_MODEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No trained v2 model found at '{EXPANDED_V2_MODEL_PATH}'. "
                "Run dataset_generator_v2.py, then train_expanded_model_v2.py."
            )
    return _model


def predict_window_event(
    window_readings: Union[pd.DataFrame, Sequence[Mapping[str, float]]],
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict the event for one window of exactly WINDOW_SIZE consecutive readings
    (each containing every key in common.FEATURE_COLUMNS, in time order).

    Returns (predicted_label, confidence, class_probabilities).
    """
    model = load_model()

    window_df = (
        window_readings if isinstance(window_readings, pd.DataFrame) else pd.DataFrame(window_readings)
    )
    if len(window_df) != WINDOW_SIZE:
        raise ValueError(f"Expected exactly {WINDOW_SIZE} readings in a window, got {len(window_df)}.")
    missing: List[str] = [c for c in FEATURE_COLUMNS if c not in window_df.columns]
    if missing:
        raise ValueError(f"window_readings is missing required keys: {missing}")

    features = extract_window_features(window_df)
    X = pd.DataFrame([features], columns=WINDOW_FEATURE_COLUMNS)

    predicted_label = model.predict(X)[0]
    class_probabilities = dict(zip(model.classes_, model.predict_proba(X)[0]))
    return predicted_label, class_probabilities[predicted_label], class_probabilities
