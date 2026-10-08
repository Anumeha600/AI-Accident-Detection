"""
Phase 4 recording-level train/test separation (train_expanded_model.py).

Two kinds of check:
  * build_windows_from_dataset() never lets a window cross a recording boundary.
  * Running the real main() (with the model output redirected to a temp file)
    exercises its own leakage assertions, and we independently verify the
    split it performs; the committed .joblib/.csv files are left untouched.
"""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import dataset_generator as dg
import train_expanded_model as tem
from common import LABEL_COLUMN
from window_features import WINDOW_FEATURE_COLUMNS


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(Path(__file__).resolve().parent.parent / tem.DATASET_CSV)


def test_dataset_shape(dataset):
    recs = dataset[["recording_id", LABEL_COLUMN]].drop_duplicates()
    assert len(recs) == 600
    assert recs["recording_id"].is_unique            # each id belongs to ONE scenario
    assert (recs[LABEL_COLUMN].value_counts() == 100).all()
    assert (dataset.groupby("recording_id").size() == 50).all()


def test_windows_never_cross_recordings():
    rng = np.random.default_rng(0)
    df = pd.concat([dg.generate_recording(rng, "Pothole", 0),
                    dg.generate_recording(rng, "Sharp Turn", 1)], ignore_index=True)
    w = tem.build_windows_from_dataset(df)
    assert len(w) == 10                                          # 5 per 50-sample recording
    assert w.groupby("recording_id").size().to_dict() == {0: 5, 1: 5}
    assert set(w.loc[w.recording_id == 0, LABEL_COLUMN]) == {"Pothole"}
    assert set(w.loc[w.recording_id == 1, LABEL_COLUMN]) == {"Sharp Turn"}
    assert list(w.loc[w.recording_id == 0, "window_start_row"]) == [0, 10, 20, 30, 40]


def test_recording_level_split_has_no_overlap_and_is_stratified(dataset):
    # Same call train_expanded_model.main() makes.
    recordings = dataset[["recording_id", LABEL_COLUMN]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        recordings["recording_id"], test_size=0.2, random_state=42,
        stratify=recordings[LABEL_COLUMN],
    )
    train_ids, test_ids = set(train_ids), set(test_ids)
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) == 480 and len(test_ids) == 120

    tw = tem.build_windows_from_dataset(dataset[dataset.recording_id.isin(train_ids)])
    sw = tem.build_windows_from_dataset(dataset[dataset.recording_id.isin(test_ids)])
    assert set(tw.recording_id).isdisjoint(set(sw.recording_id))
    assert set(tw.recording_id) == train_ids and set(sw.recording_id) == test_ids
    assert len(tw) == 480 * 5 and len(sw) == 120 * 5
    # 20 held-out recordings per class
    assert (sw.groupby(LABEL_COLUMN).recording_id.nunique() == 20).all()


@pytest.mark.slow
def test_main_runs_its_leakage_asserts_and_does_not_touch_committed_files(
        project_root, tmp_path, monkeypatch, capsys):
    protected = [project_root / tem.DATASET_CSV, project_root / tem.EXPANDED_MODEL_PATH,
                 project_root / "accident_classifier_model.joblib",
                 project_root / "accident_classifier_model_windowed.joblib"]
    before = {p: md5(p) for p in protected}

    out_model = tmp_path / "tmp_model.joblib"
    monkeypatch.setattr(tem, "EXPANDED_MODEL_PATH", str(out_model))
    tem.main()                       # raises AssertionError on any recording leakage

    out = capsys.readouterr().out
    assert "zero recording_id overlap" in out
    assert out_model.exists()
    model = joblib.load(out_model)
    assert list(model.feature_names_in_) == WINDOW_FEATURE_COLUMNS
    assert {p: md5(p) for p in protected} == before
