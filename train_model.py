"""
PHASE 2 -- Train an accident-event classifier.
------------------------------------------------
Loads the six simulated_*.csv files produced in Phase 1
(sensor_simulator.py), combines them, and trains a Random Forest
classifier to recognize the driving/accident event from sensor
readings alone.

IMPORTANT:
    This model is trained and evaluated ONLY on SIMULATED / SYNTHETIC
    data. It has NOT been validated against real-world accident data
    and must not be treated as a certified accident-detection system.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from common import FEATURE_COLUMNS, LABEL_COLUMN, MODEL_PATH, find_csv_files


def load_dataset():
    """Load and combine all simulated_*.csv files into one DataFrame."""
    csv_files = find_csv_files()
    if not csv_files:
        raise FileNotFoundError(
            "No simulated_*.csv files found in this folder. Run "
            "sensor_simulator.py (Phase 1) first to generate them."
        )

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")

    frames = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined dataset: {len(combined)} rows total.")
    print(combined[LABEL_COLUMN].value_counts().to_string())
    return combined


def show_confusion_matrix(y_test, y_pred, labels):
    """Print the confusion matrix as a clearly labeled table."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual \\ Predicted"
    print(cm_df.to_string())


def main():
    print("=" * 70)
    print("PHASE 2: Training accident-event classifier")
    print("(Trained and evaluated on SIMULATED / SYNTHETIC data only)")
    print("=" * 70 + "\n")

    df = load_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    # Stratified split: keeps the same proportion of each scenario in both
    # the training set and the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # Random Forest: a simple, interpretable model that works well on
    # small tabular datasets like this one.
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "-" * 70)
    print("EVALUATION RESULTS (held-out SIMULATED test data)")
    print("-" * 70)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}\n")

    print("Precision / Recall / F1-score per class:")
    print(classification_report(y_test, y_pred))

    labels = sorted(y.unique())
    print("Confusion Matrix (rows = actual scenario, columns = predicted scenario):")
    show_confusion_matrix(y_test, y_pred, labels)

    # Feature importance -- a useful, interpretable bonus from Random Forest.
    print("\nFeature importance (which sensors mattered most):")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).round(3).to_string())

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("REMINDER: This model was trained ONLY on simulated/synthetic")
    print("data. It is NOT validated against real-world accident data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
