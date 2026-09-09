"""
Wraps the trained RandomForest classifier + label encoder. Loaded once at
startup - joblib deserialization isn't free, and there's no reason to
repeat it per-request.
"""

import joblib
import numpy as np

from app.config import settings


class ClassifierService:

    def __init__(self):
        self.classifier = joblib.load(settings.classifier_path)
        self.label_encoder = joblib.load(settings.label_encoder_path)

    def predict(self, embedding: np.ndarray) -> tuple[str, float]:
        embedding_2d = embedding.reshape(1, -1)
        probabilities = self.classifier.predict_proba(embedding_2d)[0]

        predicted_idx = int(np.argmax(probabilities))
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx])

        return predicted_label, confidence

    @property
    def known_mechanisms(self) -> list[str]:
        return list(self.label_encoder.classes_)
