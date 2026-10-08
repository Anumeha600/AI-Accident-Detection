"""
Loading and calling the committed Phase 4 model. These tests check the
INTERFACE (shape, types, validation), not accuracy -- model quality is not
asserted here, and nothing is retrained or modified.
"""

import hashlib

import numpy as np
import pytest

import dataset_generator as dg
import predict_expanded
from common import FEATURE_COLUMNS
from window_features import WINDOW_FEATURE_COLUMNS, WINDOW_SIZE

SCENARIOS = ["Hard Braking", "Minor Accident", "Normal Driving", "Pothole",
             "Severe Accident", "Sharp Turn"]


def window_for(scenario, seed=0):
    rec = dg.generate_recording(np.random.default_rng(seed), scenario, 0)
    return rec.iloc[:WINDOW_SIZE][FEATURE_COLUMNS]


def test_model_loads_and_is_cached():
    m = predict_expanded.load_model()
    assert m is predict_expanded.load_model()


def test_model_schema():
    m = predict_expanded.load_model()
    assert sorted(m.classes_) == SCENARIOS
    assert m.n_features_in_ == len(WINDOW_FEATURE_COLUMNS) == 36
    assert list(m.feature_names_in_) == WINDOW_FEATURE_COLUMNS


def test_missing_model_file_gives_helpful_error(monkeypatch):
    monkeypatch.setattr(predict_expanded, "_model", None)
    monkeypatch.setattr(predict_expanded, "EXPANDED_MODEL_PATH", "no_such_model.joblib")
    with pytest.raises(FileNotFoundError, match="train_expanded_model.py"):
        predict_expanded.load_model()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_predict_returns_valid_label_confidence_and_probabilities(scenario):
    label, conf, probs = predict_expanded.predict_window_event(window_for(scenario))
    assert label in SCENARIOS
    assert set(probs) == set(SCENARIOS)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert conf == probs[label] == max(probs.values())


def test_predict_accepts_list_of_dicts_like_the_live_pipeline():
    df = window_for("Normal Driving")
    as_dicts = df.to_dict("records")
    assert predict_expanded.predict_window_event(as_dicts)[0] == predict_expanded.predict_window_event(df)[0]


def test_predict_is_deterministic():
    w = window_for("Hard Braking", seed=3)
    assert predict_expanded.predict_window_event(w) == predict_expanded.predict_window_event(w)


def test_wrong_window_length_rejected():
    with pytest.raises(ValueError, match="exactly"):
        predict_expanded.predict_window_event(window_for("Pothole").iloc[:5])


def test_missing_feature_key_rejected():
    with pytest.raises(ValueError, match="missing"):
        predict_expanded.predict_window_event(window_for("Pothole").drop(columns=["speed_kmh"]))


def test_prediction_does_not_modify_model_file(project_root):
    path = project_root / predict_expanded.EXPANDED_MODEL_PATH
    before = hashlib.md5(path.read_bytes()).hexdigest()
    predict_expanded.predict_window_event(window_for("Severe Accident"))
    assert hashlib.md5(path.read_bytes()).hexdigest() == before
