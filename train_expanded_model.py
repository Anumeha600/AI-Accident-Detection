"""
PHASE 4 -- Window-based classifier trained on the EXPANDED synthetic
dataset (many independent recordings per scenario), with a train/test
split done at the RECORDING level to prevent leakage.
----------------------------------------------------------------------
Phase 3 windowed a single 5-second recording per scenario (30 windows
total, 6 in the test set) -- too small to trust. This script uses the
600-recording dataset from dataset_generator.py (Phase 4) instead, and
reuses the exact same windowing/feature-extraction code from Phase 3
(window_features.py) for a like-for-like comparison.

WHY SPLIT BY RECORDING, NOT BY WINDOW:
    If windows were split randomly, two windows from the SAME recording
    (which share the same noise realization, the same event, and highly
    similar values) could end up on both sides of the split. The model
    could then partly "recognize" a specific recording rather than
    learning to generalize across recordings -- inflating the score.
    Splitting on recording_id first guarantees every recording's windows
    stay entirely in train or entirely in test.

IMPORTANT:
    - This is a SEPARATE, EXPERIMENTAL pipeline. It does NOT modify or
      overwrite any Phase 2 or Phase 3 file or model.
    - Trained and evaluated ONLY on SIMULATED / SYNTHETIC data.
    - Results here do NOT represent real-world accident-detection
      performance -- see the limitations printed at the end.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from common import LABEL_COLUMN
from window_features import WINDOW_SIZE, STRIDE, WINDOW_FEATURE_COLUMNS, extract_window_features, make_windows

DATASET_CSV = "expanded_synthetic_dataset.csv"
EXPANDED_MODEL_PATH = "accident_classifier_model_expanded.joblib"

# Reference numbers already reported for Phase 2 (row-based, single
# recording/scenario) and Phase 3 (window-based, single recording/
# scenario). Re-run train_model.py / train_window_model.py to reconfirm
# -- printed here, unmodified, purely for a side-by-side comparison.
PHASE2_ACCURACY = 0.750
PHASE2_F1_BY_CLASS = {
    "Hard Braking": 1.00, "Minor Accident": 0.78, "Normal Driving": 0.48,
    "Pothole": 0.53, "Severe Accident": 0.90, "Sharp Turn": 0.82,
}
PHASE3_ACCURACY = 0.833
PHASE3_F1_BY_CLASS = {
    "Hard Braking": 0.67, "Minor Accident": 1.00, "Normal Driving": 0.00,
    "Pothole": 1.00, "Severe Accident": 1.00, "Sharp Turn": 1.00,
}


def build_windows_from_dataset(df, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Window each recording SEPARATELY (grouped by recording_id) using the
    same make_windows/extract_window_features functions Phase 3 uses, so
    windows never cross a recording boundary.
    """
    rows = []
    for recording_id, group in df.groupby("recording_id", sort=False):
        group = group.sort_values("sample_index").reset_index(drop=True)
        scenario = group[LABEL_COLUMN].iloc[0]
        starts = range(0, len(group), stride)
        for window_start, window_df in zip(starts, make_windows(group, window_size, stride)):
            feats = extract_window_features(window_df)
            feats[LABEL_COLUMN] = scenario
            feats["recording_id"] = recording_id
            feats["window_start_row"] = window_start
            rows.append(feats)
    return pd.DataFrame(rows)


def show_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual \\ Predicted"
    print(cm_df.to_string())


def main():
    print("=" * 70)
    print("PHASE 4: Window-based classifier on the EXPANDED dataset")
    print("(Trained and evaluated on SIMULATED / SYNTHETIC data only)")
    print("=" * 70 + "\n")

    try:
        df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{DATASET_CSV}' not found. Run dataset_generator.py first."
        )

    recordings = df[["recording_id", LABEL_COLUMN]].drop_duplicates()
    print(f"Total recordings: {len(recordings)}  |  Total rows: {len(df)}")
    print("\nRecordings per scenario:")
    print(recordings[LABEL_COLUMN].value_counts().sort_index().to_string())

    # --- Split at the RECORDING level, BEFORE windowing ---
    train_ids, test_ids = train_test_split(
        recordings["recording_id"],
        test_size=0.2,
        random_state=42,
        stratify=recordings[LABEL_COLUMN],
    )
    train_ids, test_ids = set(train_ids), set(test_ids)
    assert train_ids.isdisjoint(test_ids), "Recording IDs leaked between train and test!"

    train_df = df[df["recording_id"].isin(train_ids)]
    test_df = df[df["recording_id"].isin(test_ids)]
    print(f"\nTrain recordings: {len(train_ids)}  |  Test recordings: {len(test_ids)}")
    print("(Confirmed: zero recording_id overlap between train and test.)")

    # --- Window each side SEPARATELY so no window can cross the split ---
    print(f"\nWindowing with window_size={WINDOW_SIZE}, stride={STRIDE} "
          "(same as Phase 3, non-overlapping)...")
    train_windows = build_windows_from_dataset(train_df)
    test_windows = build_windows_from_dataset(test_df)

    shared_recordings = set(train_windows["recording_id"]) & set(test_windows["recording_id"])
    assert not shared_recordings, "A recording produced windows on both sides!"

    print(f"\nTraining windows: {len(train_windows)}  |  Test windows: {len(test_windows)}")
    print("\nTraining windows per class:")
    print(train_windows[LABEL_COLUMN].value_counts().sort_index().to_string())
    print("\nTest windows per class:")
    print(test_windows[LABEL_COLUMN].value_counts().sort_index().to_string())

    X_train = train_windows[WINDOW_FEATURE_COLUMNS]
    y_train = train_windows[LABEL_COLUMN]
    X_test = test_windows[WINDOW_FEATURE_COLUMNS]
    y_test = test_windows[LABEL_COLUMN]

    # Same model type/hyperparameters as Phase 2 and Phase 3, for a
    # like-for-like comparison -- not tuned for a higher score.
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "-" * 70)
    print("EVALUATION RESULTS -- expanded window-based model (SIMULATED test data)")
    print("-" * 70)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}\n")

    print("Precision / Recall / F1-score per class:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y_train.unique())
    print("Confusion Matrix (rows = actual scenario, columns = predicted scenario):")
    show_confusion_matrix(y_test, y_pred, labels)

    print(f"\nFeature importance (all {len(WINDOW_FEATURE_COLUMNS)} features, sorted):")
    importances = pd.Series(model.feature_importances_, index=WINDOW_FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).round(3).to_string())

    joblib.dump(model, EXPANDED_MODEL_PATH)
    print(f"\nExpanded-dataset model saved to: {EXPANDED_MODEL_PATH}")
    print("(Phase 2 and Phase 3 models were NOT modified.)")

    print("\n" + "=" * 70)
    print("COMPARISON: Phase 2 (row) vs Phase 3 (window, 1 recording/class)")
    print("            vs Phase 4 (window, 100 recordings/class)")
    print("=" * 70)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(f"{'Scenario':16s} {'Phase2 F1':>10s} {'Phase3 F1':>10s} {'Phase4 F1':>10s}")
    for scenario in sorted(PHASE2_F1_BY_CLASS):
        p4_f1 = report_dict.get(scenario, {}).get("f1-score", float("nan"))
        print(f"{scenario:16s} {PHASE2_F1_BY_CLASS[scenario]:>10.2f} "
              f"{PHASE3_F1_BY_CLASS[scenario]:>10.2f} {p4_f1:>10.2f}")
    print(f"{'ACCURACY':16s} {PHASE2_ACCURACY:>10.2f} {PHASE3_ACCURACY:>10.2f} {accuracy:>10.2f}")

    print("\nNOTE: Phase 4 has ~100x more test windows than Phase 3, from 20")
    print("independent held-out recordings per class rather than 1 -- these")
    print("numbers are far more statistically trustworthy than Phase 3's,")
    print("though still only a comparison between two synthetic-data setups.")

    print("\n" + "=" * 70)
    print("LIMITATIONS OF THIS SYNTHETIC EXPERIMENT")
    print("=" * 70)
    print(
        "- All data (Phase 1-4) is simulated/synthetic; no real accelerometer,\n"
        "  gyroscope, vibration, or speed sensor was involved.\n"
        "- Recordings are generated by a hand-written parametric model (noise\n"
        "  + a smooth event bump/transition), not recorded from real driving,\n"
        "  so they cannot capture the full complexity, sensor artifacts, or\n"
        "  unpredictability of real-world driving and real accidents.\n"
        "- Randomization ranges (event timing/intensity/width, baseline speed)\n"
        "  were chosen by hand to be 'reasonable', not derived from real\n"
        "  telemetry -- they encode the author's assumptions, not measured\n"
        "  physics.\n"
        "- A higher accuracy here than Phase 2/3 does NOT prove this pipeline\n"
        "  would work on a real vehicle -- it only shows the model can\n"
        "  generalize across independent synthetic recordings produced by the\n"
        "  SAME generator, which is a much lower bar than real-world\n"
        "  generalization.\n"
        "- These results must not be cited as evidence of real-world\n"
        "  accident-detection performance."
    )


if __name__ == "__main__":
    main()
