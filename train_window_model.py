"""
PHASE 3 (EXPERIMENTAL) -- Window-based accident-event classifier.
--------------------------------------------------------------------
Phase 2 (train_model.py) classifies individual sensor ROWS and mainly
confuses Normal Driving with Pothole -- a pothole is a brief event, so
most of its individual rows look identical to normal driving. This
experiment instead classifies a short WINDOW of consecutive rows using
statistical features computed over each window (see window_features.py),
to test whether temporal context helps.

IMPORTANT:
    - This is a SEPARATE, EXPERIMENTAL pipeline. It does NOT modify or
      overwrite any Phase 2 file (common.py, train_model.py, predict.py,
      or accident_classifier_model.joblib).
    - Trained and evaluated ONLY on SIMULATED / SYNTHETIC data.
    - A higher (or lower) score here does NOT prove this approach is
      better for real-world accident detection -- see the limitations
      printed at the end of this script's output.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from common import LABEL_COLUMN, find_csv_files
from window_features import (
    WINDOW_SIZE,
    STRIDE,
    WINDOW_FEATURE_COLUMNS,
    build_windowed_dataset,
)

# Deliberately different from Phase 2's MODEL_PATH -- the Phase 2 model
# file is never touched by this script.
WINDOW_MODEL_PATH = "accident_classifier_model_windowed.joblib"

# Reference numbers from the Phase 2 row-based run (train_model.py), as
# already reported. Re-run train_model.py yourself to reconfirm them --
# they are only printed here, unmodified, for a side-by-side comparison.
PHASE2_ACCURACY = 0.750
PHASE2_F1_BY_CLASS = {
    "Hard Braking": 1.00,
    "Minor Accident": 0.78,
    "Normal Driving": 0.48,
    "Pothole": 0.53,
    "Severe Accident": 0.90,
    "Sharp Turn": 0.82,
}


def show_confusion_matrix(y_test, y_pred, labels):
    """Print the confusion matrix as a clearly labeled table."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual \\ Predicted"
    print(cm_df.to_string())


def main():
    print("=" * 70)
    print("PHASE 3 (EXPERIMENTAL): Window-based accident-event classifier")
    print("(Trained and evaluated on SIMULATED / SYNTHETIC data only)")
    print("=" * 70)
    print(f"\nWindow size: {WINDOW_SIZE} samples ({WINDOW_SIZE / 10:.1f} s at 10 Hz)")
    print(f"Stride: {STRIDE} samples (non-overlapping -- no raw row is reused")
    print("across windows, which avoids leaking near-duplicate data between")
    print("the train and test sets).\n")

    csv_files = find_csv_files()
    if not csv_files:
        raise FileNotFoundError(
            "No simulated_*.csv files found. Run sensor_simulator.py first."
        )
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")

    dataset = build_windowed_dataset(csv_files)
    print(f"\nTotal windows generated: {len(dataset)}")
    print("Windows per scenario:")
    print(dataset[LABEL_COLUMN].value_counts().sort_index().to_string())

    X = dataset[WINDOW_FEATURE_COLUMNS]
    y = dataset[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining windows: {len(X_train)}  |  Test windows: {len(X_test)}")
    print("\nTraining set counts per class:")
    print(y_train.value_counts().sort_index().to_string())
    print("\nTest set counts per class:")
    print(y_test.value_counts().sort_index().to_string())

    # Same model type/hyperparameters as Phase 2, for a like-for-like
    # comparison -- this is NOT an attempt to tune for a higher score.
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "-" * 70)
    print("EVALUATION RESULTS -- window-based model (SIMULATED test data)")
    print("-" * 70)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}\n")

    print("Precision / Recall / F1-score per class:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y.unique())
    print("Confusion Matrix (rows = actual scenario, columns = predicted scenario):")
    show_confusion_matrix(y_test, y_pred, labels)

    print(f"\nFeature importance (all {len(WINDOW_FEATURE_COLUMNS)} features, sorted):")
    importances = pd.Series(model.feature_importances_, index=WINDOW_FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).round(3).to_string())

    joblib.dump(model, WINDOW_MODEL_PATH)
    print(f"\nWindow-based model saved to: {WINDOW_MODEL_PATH}")
    print("(Phase 2's accident_classifier_model.joblib was NOT modified.)")

    print("\n" + "=" * 70)
    print("COMPARISON: Phase 2 (row-based) vs Phase 3 (window-based)")
    print("=" * 70)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(f"{'Scenario':16s} {'Phase2 F1':>10s} {'Phase3 F1':>10s}")
    for scenario in sorted(PHASE2_F1_BY_CLASS):
        p3_f1 = report_dict.get(scenario, {}).get("f1-score", float("nan"))
        print(f"{scenario:16s} {PHASE2_F1_BY_CLASS[scenario]:>10.2f} {p3_f1:>10.2f}")
    print(f"{'ACCURACY':16s} {PHASE2_ACCURACY:>10.2f} {accuracy:>10.2f}")

    print("\nNOTE: This comparison is illustrative only -- Phase 3 has far")
    print("fewer samples (windows) than Phase 2 (rows), all drawn from a")
    print("single synthetic recording per scenario. See limitations below.")

    print("\n" + "=" * 70)
    print("LIMITATIONS OF THIS SYNTHETIC EXPERIMENT")
    print("=" * 70)
    print(
        "- All data (Phase 1-3) is simulated/synthetic; no real accelerometer,\n"
        "  gyroscope, vibration, or speed sensor was involved.\n"
        "- Each scenario has only ONE 5-second synthetic recording (50 rows),\n"
        "  so windowing it yields only a handful of windows per class. This\n"
        "  makes the Phase 3 test set very small and its metrics high-variance\n"
        "  -- a good or bad score here is not statistically robust.\n"
        "- Every simulated event is centered at the same sample index in its\n"
        "  recording, so windows line up neatly with 'event' vs 'background'\n"
        "  in a way real-world driving data would not. This may make window\n"
        "  classification look cleaner than it would be in practice.\n"
        "- A higher (or lower) accuracy than Phase 2 does NOT prove window-\n"
        "  based classification is better (or worse) for real-world accident\n"
        "  detection -- both models are trained and tested only on synthetic\n"
        "  data generated by the same simulator."
    )


if __name__ == "__main__":
    main()
