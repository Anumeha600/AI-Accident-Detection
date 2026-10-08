"""
PHASE 7 -- Window-based classifier on the 8-scenario synthetic dataset (v2).
-----------------------------------------------------------------------------
Follows the Phase 4 methodology (train_expanded_model.py) exactly, on
expanded_synthetic_dataset_v2.csv (800 recordings, 8 scenarios):

    1. Stratified 80/20 split BY recording_id, BEFORE windowing.
    2. Window each side separately (10-sample, non-overlapping windows).
    3. The 36 features from window_features.py, unchanged.
    4. RandomForestClassifier(n_estimators=200, random_state=42).

Phase 4's windowing function is imported (not copied) so both phases window
identically. Nothing from Phase 2-4 is modified: the model is written to a
NEW file (accident_classifier_model_expanded_v2.joblib) and the evaluation to
a NEW JSON report. The Phase 4 baseline printed for comparison is re-trained
IN MEMORY from expanded_synthetic_dataset.csv with Phase 4's exact settings;
it is never saved.

IMPORTANT: Everything here is trained and evaluated ONLY on SIMULATED /
SYNTHETIC data. The numbers say how well a model separates synthetic
recordings from the same hand-written generator; they are NOT an estimate of
real-world accident-detection performance, and the Phase 4 vs Phase 7
comparison is between two synthetic experiments with different class sets.
"""

import json
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from common import LABEL_COLUMN
from window_features import WINDOW_FEATURE_COLUMNS, WINDOW_SIZE, STRIDE
from train_expanded_model import build_windows_from_dataset  # shared Phase 4 windowing

DATASET_CSV_V2 = "expanded_synthetic_dataset_v2.csv"
BASELINE_CSV_PHASE4 = "expanded_synthetic_dataset.csv"   # read-only, for the baseline
EXPANDED_V2_MODEL_PATH = "accident_classifier_model_expanded_v2.joblib"
EVALUATION_REPORT_PATH = "evaluation_report_expanded_v2.json"

TEST_SIZE = 0.2
SPLIT_SEED = 42
MODEL_PARAMS = dict(n_estimators=200, random_state=42)   # same as Phase 2-4


def split_by_recording(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified recording-level split, BEFORE windowing. Returns (train_df, test_df)."""
    recordings = df[["recording_id", LABEL_COLUMN]].drop_duplicates()
    assert recordings["recording_id"].is_unique, "a recording_id has more than one scenario label"

    train_ids, test_ids = train_test_split(
        recordings["recording_id"], test_size=TEST_SIZE, random_state=SPLIT_SEED,
        stratify=recordings[LABEL_COLUMN],
    )
    train_ids, test_ids = set(train_ids), set(test_ids)
    assert train_ids.isdisjoint(test_ids), "Recording IDs leaked between train and test!"
    return df[df["recording_id"].isin(train_ids)], df[df["recording_id"].isin(test_ids)]


def window_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Window each side SEPARATELY and verify no recording appears on both sides."""
    train_windows = build_windows_from_dataset(train_df)
    test_windows = build_windows_from_dataset(test_df)
    shared = set(train_windows["recording_id"]) & set(test_windows["recording_id"])
    assert not shared, f"Recordings produced windows on both sides: {sorted(shared)[:5]}"
    return train_windows, test_windows


def phase4_baseline() -> Dict[str, float]:
    """
    Re-run Phase 4's recipe IN MEMORY on the Phase 4 (6-scenario) dataset to
    get its synthetic test accuracy / macro-F1. Nothing is saved or modified.
    """
    df = pd.read_csv(BASELINE_CSV_PHASE4)
    train_df, test_df = split_by_recording(df)
    train_w, test_w = window_split(train_df, test_df)
    model = RandomForestClassifier(**MODEL_PARAMS).fit(train_w[WINDOW_FEATURE_COLUMNS], train_w[LABEL_COLUMN])
    pred = model.predict(test_w[WINDOW_FEATURE_COLUMNS])
    _, _, f1, _ = precision_recall_fscore_support(test_w[LABEL_COLUMN], pred, average="macro", zero_division=0)
    return {
        "n_classes": int(test_w[LABEL_COLUMN].nunique()),
        "n_test_windows": int(len(test_w)),
        "accuracy": float(accuracy_score(test_w[LABEL_COLUMN], pred)),
        "macro_f1": float(f1),
    }


def main() -> None:
    print("=" * 70)
    print("PHASE 7: Window-based classifier on the 8-scenario EXPANDED dataset (v2)")
    print("(Trained and evaluated on SIMULATED / SYNTHETIC data only)")
    print("=" * 70 + "\n")

    try:
        df = pd.read_csv(DATASET_CSV_V2)
    except FileNotFoundError:
        raise FileNotFoundError(f"'{DATASET_CSV_V2}' not found. Run dataset_generator_v2.py first.")

    recordings = df[["recording_id", LABEL_COLUMN]].drop_duplicates()
    samples_per_recording = df.groupby("recording_id").size()
    print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Total recordings: {len(recordings)}  |  samples per recording: "
          f"{[int(x) for x in sorted(samples_per_recording.unique())]}")
    print("\nRecordings per scenario:")
    print(recordings[LABEL_COLUMN].value_counts().sort_index().to_string())

    # --- Split at the RECORDING level, BEFORE windowing ---
    train_df, test_df = split_by_recording(df)
    train_ids, test_ids = set(train_df["recording_id"]), set(test_df["recording_id"])
    print(f"\nTrain recordings: {len(train_ids)}  |  Test recordings: {len(test_ids)}")
    print("(Confirmed: zero recording_id overlap between train and test.)")

    train_windows, test_windows = window_split(train_df, test_df)
    print(f"\nWindowing: window_size={WINDOW_SIZE}, stride={STRIDE} (non-overlapping), "
          f"{len(WINDOW_FEATURE_COLUMNS)} features per window")
    print(f"Training windows: {len(train_windows)}  |  Test windows: {len(test_windows)}  "
          f"|  total: {len(train_windows) + len(test_windows)}")
    print("(Confirmed: no recording contributes windows to both train and test.)")
    print("\nTraining windows per class:")
    print(train_windows[LABEL_COLUMN].value_counts().sort_index().to_string())
    print("\nTest windows per class:")
    print(test_windows[LABEL_COLUMN].value_counts().sort_index().to_string())

    X_train, y_train = train_windows[WINDOW_FEATURE_COLUMNS], train_windows[LABEL_COLUMN]
    X_test, y_test = test_windows[WINDOW_FEATURE_COLUMNS], test_windows[LABEL_COLUMN]

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = sorted(y_train.unique())
    accuracy = accuracy_score(y_test, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "-" * 70)
    print("EVALUATION RESULTS -- 8-class window model (SIMULATED test data)")
    print("-" * 70)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro    precision/recall/F1: {macro[0]:.3f} / {macro[1]:.3f} / {macro[2]:.3f}")
    print(f"Weighted precision/recall/F1: {weighted[0]:.3f} / {weighted[1]:.3f} / {weighted[2]:.3f}\n")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual \\ Predicted"
    print("Confusion Matrix (rows = actual scenario, columns = predicted scenario):")
    print(cm_df.to_string())

    importances = pd.Series(model.feature_importances_, index=WINDOW_FEATURE_COLUMNS).sort_values(ascending=False)
    print(f"\nFeature importance (all {len(importances)} features, sorted):")
    print(importances.round(3).to_string())

    joblib.dump(model, EXPANDED_V2_MODEL_PATH)
    print(f"\nPhase 7 model saved to: {EXPANDED_V2_MODEL_PATH}")
    print("(No Phase 2/3/4 model or dataset was modified.)")

    baseline = phase4_baseline()
    print("\n" + "=" * 70)
    print("SYNTHETIC-vs-SYNTHETIC COMPARISON (not real-world performance)")
    print("=" * 70)
    print(f"{'Experiment':34s} {'classes':>7s} {'test windows':>13s} {'accuracy':>9s} {'macro F1':>9s}")
    print(f"{'Phase 4 (6 scenarios, re-run)':34s} {baseline['n_classes']:>7d} "
          f"{baseline['n_test_windows']:>13d} {baseline['accuracy']:>9.3f} {baseline['macro_f1']:>9.3f}")
    print(f"{'Phase 7 (8 scenarios)':34s} {len(labels):>7d} {len(test_windows):>13d} "
          f"{accuracy:>9.3f} {macro[2]:>9.3f}")
    print("Different class sets and test sets: the two rows are not like-for-like.")

    report = {
        "disclaimer": ("All data is SIMULATED/SYNTHETIC. These figures do not measure real-world "
                       "accident-detection performance."),
        "dataset_csv": DATASET_CSV_V2,
        "model_path": EXPANDED_V2_MODEL_PATH,
        "dataset_rows": int(len(df)),
        "n_recordings": int(len(recordings)),
        "samples_per_recording": [int(x) for x in sorted(samples_per_recording.unique())],
        "recordings_per_class": {k: int(v) for k, v in recordings[LABEL_COLUMN].value_counts().sort_index().items()},
        "split": {"test_size": TEST_SIZE, "random_state": SPLIT_SEED, "stratified": True,
                  "level": "recording_id, before windowing",
                  "train_recordings": len(train_ids), "test_recordings": len(test_ids),
                  "recording_id_overlap": len(train_ids & test_ids)},
        "windowing": {"window_size": WINDOW_SIZE, "stride": STRIDE, "n_features": len(WINDOW_FEATURE_COLUMNS),
                      "train_windows": int(len(train_windows)), "test_windows": int(len(test_windows))},
        "model": {"type": "RandomForestClassifier", **MODEL_PARAMS},
        "accuracy": float(accuracy),
        "macro": dict(zip(["precision", "recall", "f1"], map(float, macro[:3]))),
        "weighted": dict(zip(["precision", "recall", "f1"], map(float, weighted[:3]))),
        "per_class": {lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for lab, p, r, f, s in zip(labels, prec, rec, f1, support)},
        "confusion_matrix": {"labels": labels, "rows_actual_cols_predicted": cm.tolist()},
        "feature_importance": {k: float(v) for k, v in importances.items()},
        "phase4_baseline_synthetic_rerun_in_memory": baseline,
    }
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nEvaluation report saved to: {EVALUATION_REPORT_PATH}")

    print("\n" + "=" * 70)
    print("LIMITATIONS OF THIS SYNTHETIC EXPERIMENT")
    print("=" * 70)
    print(
        "- All data (Phases 1-7) is simulated; no real sensor or vehicle was involved.\n"
        "- The Rollover and Multi-Impact scenarios are hand-written, simplified models\n"
        "  (see dataset_generator_v2.py); their signatures reflect the author's\n"
        "  assumptions, not measured crashes.\n"
        "- Test recordings come from the SAME generator as training recordings, so the\n"
        "  scores show separability of synthetic scenarios only.\n"
        "- Each window carries its recording's label, even windows from before an event\n"
        "  starts; this label noise is inherited from Phase 4's methodology.\n"
        "- Do not cite these results as real-world accident-detection accuracy."
    )


if __name__ == "__main__":
    main()
