"""
Trains a classifier head on top of cached ESM2 embeddings to predict
resistance mechanism, and separately fits a novelty detector: per-class
centroids in embedding space, so a new sequence's distance to its
nearest known centroid can be used as a "how well does this match known
resistance patterns" signal at inference time.

Evaluates with per-class precision/recall/f1, not just overall accuracy -
the training data is heavily imbalanced (one class is ~87% of examples),
so accuracy alone would hide poor performance on rarer mechanism classes.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

EMBEDDINGS_PATH = "data/processed/embeddings.npy"
LABELS_PATH = "data/processed/labels.csv"

MODEL_OUTPUT_PATH = "data/processed/classifier.joblib"
LABEL_ENCODER_PATH = "data/processed/label_encoder.joblib"
CENTROIDS_PATH = "data/processed/class_centroids.joblib"
METRICS_PATH = "data/processed/training_metrics.json"


def train() -> None:
    X = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    y_raw = labels_df["resistance_mechanism"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # class_weight="balanced" matters a lot here given the ~87%-majority-class
    # imbalance - without it, the classifier would likely just predict the
    # majority class every time and still show high accuracy.
    classifier = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
    )

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    # Novelty detection: compute the centroid (mean embedding) of each known
    # resistance mechanism class, using the FULL dataset (not just train
    # split) so novelty scoring at inference time is judged against the
    # most complete picture of "what known resistance looks like".
    centroids = {}
    for class_idx in np.unique(y):
        class_name = label_encoder.inverse_transform([class_idx])[0]
        centroids[class_name] = X[y == class_idx].mean(axis=0)

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(classifier, MODEL_OUTPUT_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(centroids, CENTROIDS_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train()
