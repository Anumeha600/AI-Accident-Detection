"""Phase 7: recording-level separation, windowing, v2 model loading and prediction."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import dataset_generator_v2 as g2
import predict_expanded_v2 as p2
import train_expanded_model as tem
import train_expanded_model_v2 as t2
from common import FEATURE_COLUMNS, LABEL_COLUMN
from window_features import WINDOW_FEATURE_COLUMNS, WINDOW_SIZE

ROOT = Path(__file__).resolve().parent.parent
EIGHT = sorted(g2.SCENARIOS_V2)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(ROOT / g2.OUTPUT_CSV_V2)


# --- recording-level separation -------------------------------------------------

def test_split_is_disjoint_stratified_and_complete(dataset):
    train_df, test_df = t2.split_by_recording(dataset)
    tr, te = set(train_df.recording_id), set(test_df.recording_id)
    assert tr.isdisjoint(te)
    assert len(tr) == 640 and len(te) == 160 and tr | te == set(dataset.recording_id)
    labels = dataset[["recording_id", LABEL_COLUMN]].drop_duplicates().set_index("recording_id")[LABEL_COLUMN]
    assert (labels.loc[sorted(te)].value_counts() == 20).all()     # 20 held-out recordings per class
    assert (labels.loc[sorted(tr)].value_counts() == 80).all()


def test_split_is_deterministic(dataset):
    a = t2.split_by_recording(dataset)[1].recording_id.unique()
    b = t2.split_by_recording(dataset)[1].recording_id.unique()
    assert list(a) == list(b)


def test_no_window_crosses_recordings_or_sides(dataset):
    train_df, test_df = t2.split_by_recording(dataset)
    tw, sw = t2.window_split(train_df, test_df)
    assert set(tw.recording_id).isdisjoint(set(sw.recording_id))
    assert len(tw) == 640 * 5 and len(sw) == 160 * 5
    for w in (tw, sw):
        assert (w.groupby("recording_id").size() == 5).all()
        assert (w.groupby("recording_id")[LABEL_COLUMN].nunique() == 1).all()
        assert list(w[WINDOW_FEATURE_COLUMNS].columns) == WINDOW_FEATURE_COLUMNS      # the unchanged 36 features
    assert len(WINDOW_FEATURE_COLUMNS) == 36


def test_window_features_come_from_exactly_one_recording():
    rng = np.random.default_rng(0)
    df = pd.concat([g2.generate_recording(rng, "Rollover", 0), g2.generate_recording(rng, "Pothole", 1)],
                   ignore_index=True)
    w = tem.build_windows_from_dataset(df)
    assert w.groupby("recording_id").size().to_dict() == {0: 5, 1: 5}
    # rollover window must have used only rollover rows: its roll rate range is huge, pothole's tiny
    assert w[w.recording_id == 0]["gyro_x_dps_range"].max() > 100
    assert w[w.recording_id == 1]["gyro_x_dps_range"].max() < 100


def test_split_assertion_fires_if_labels_conflict(dataset):
    bad = dataset.copy()
    bad.loc[bad.index[0], LABEL_COLUMN] = "Rollover"       # one recording_id, two labels
    with pytest.raises(AssertionError):
        t2.split_by_recording(bad)


# --- v2 model -----------------------------------------------------------------------

def test_v2_model_file_is_new_and_distinct():
    assert t2.EXPANDED_V2_MODEL_PATH == "accident_classifier_model_expanded_v2.joblib"
    assert t2.EXPANDED_V2_MODEL_PATH != tem.EXPANDED_MODEL_PATH
    assert (ROOT / t2.EXPANDED_V2_MODEL_PATH).exists()


def test_v2_model_has_exactly_eight_classes_and_36_features():
    m = p2.load_model()
    assert m is p2.load_model()
    assert sorted(m.classes_) == EIGHT and len(m.classes_) == 8
    assert m.n_features_in_ == 36
    assert list(m.feature_names_in_) == WINDOW_FEATURE_COLUMNS
    assert m.n_estimators == 200 and m.random_state == 42


def test_v2_wrapper_loads_only_the_v2_model(monkeypatch):
    seen = []
    real = joblib.load
    monkeypatch.setattr(p2, "_model", None)
    monkeypatch.setattr(p2.joblib, "load", lambda path, *a, **k: (seen.append(path), real(path, *a, **k))[1])
    p2.load_model()
    assert seen == ["accident_classifier_model_expanded_v2.joblib"]


def test_missing_v2_model_gives_helpful_error(monkeypatch):
    monkeypatch.setattr(p2, "_model", None)
    monkeypatch.setattr(p2, "EXPANDED_V2_MODEL_PATH", "nope.joblib")
    with pytest.raises(FileNotFoundError, match="train_expanded_model_v2.py"):
        p2.load_model()


def window_for(scenario, seed=0, start=20):
    rec = g2.generate_recording(np.random.default_rng(seed), scenario, 0)
    return rec.iloc[start:start + WINDOW_SIZE][FEATURE_COLUMNS]


@pytest.mark.parametrize("scenario", g2.SCENARIOS_V2)
def test_prediction_output_is_valid(scenario):
    label, conf, probs = p2.predict_window_event(window_for(scenario))
    assert label in EIGHT
    assert set(probs) == set(EIGHT)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert conf == probs[label] == max(probs.values())


def test_prediction_accepts_list_of_dicts_and_is_deterministic():
    df = window_for("Rollover")
    assert p2.predict_window_event(df.to_dict("records"))[0] == p2.predict_window_event(df)[0]
    assert p2.predict_window_event(df) == p2.predict_window_event(df)


def test_prediction_input_validation():
    with pytest.raises(ValueError, match="exactly"):
        p2.predict_window_event(window_for("Pothole").iloc[:4])
    with pytest.raises(ValueError, match="missing"):
        p2.predict_window_event(window_for("Pothole").drop(columns=["speed_kmh"]))


def test_saved_evaluation_report_is_consistent_with_the_dataset(dataset):
    rep = json.loads((ROOT / t2.EVALUATION_REPORT_PATH).read_text())
    assert "SYNTHETIC" in rep["disclaimer"]
    assert rep["split"]["recording_id_overlap"] == 0
    assert rep["split"]["train_recordings"] == 640 and rep["split"]["test_recordings"] == 160
    assert rep["windowing"]["train_windows"] == 3200 and rep["windowing"]["test_windows"] == 800
    assert sorted(rep["per_class"]) == EIGHT
    cm = np.array(rep["confusion_matrix"]["rows_actual_cols_predicted"])
    assert cm.sum() == 800
    assert rep["accuracy"] == pytest.approx(np.trace(cm) / cm.sum())


# --- full training run (redirected outputs) -----------------------------------------

@pytest.mark.slow
def test_main_retrains_reproducibly_without_touching_old_or_committed_files(tmp_path, monkeypatch, capsys):
    protected = [*ROOT.glob("*.joblib"), *ROOT.glob("*.csv"), ROOT / t2.EVALUATION_REPORT_PATH]
    assert len(protected) >= 11
    before = {p: md5(p) for p in protected}

    out_model, out_report = tmp_path / "m.joblib", tmp_path / "r.json"
    monkeypatch.setattr(t2, "EXPANDED_V2_MODEL_PATH", str(out_model))
    monkeypatch.setattr(t2, "EVALUATION_REPORT_PATH", str(out_report))
    t2.main()

    out = capsys.readouterr().out
    assert "zero recording_id overlap" in out and "no recording contributes windows to both" in out
    new_rep = json.loads(out_report.read_text())
    old_rep = json.loads((ROOT / "evaluation_report_expanded_v2.json").read_text())
    new_rep.pop("model_path"), old_rep.pop("model_path")      # differs only because the output was redirected
    assert new_rep == old_rep                                   # fixed seeds -> same result
    assert sorted(joblib.load(out_model).classes_) == EIGHT
    assert {p: md5(p) for p in protected} == before
