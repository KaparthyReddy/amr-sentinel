"""
Flags sequences whose embedding is far from every known resistance
mechanism's centroid as "potentially novel" - this is the actual
surveillance function of the system: emerging resistance mechanisms
won't match any existing category well, so distance-from-everything-known
is a more honest signal than classifier confidence alone (a classifier
can be confidently wrong on something genuinely new, since softmax
probabilities don't know what they don't know).
"""

import joblib
import numpy as np

from app.config import settings


class NoveltyDetector:

    def __init__(self, distance_threshold: float | None = None):
        self.centroids: dict[str, np.ndarray] = joblib.load(settings.centroids_path)
        self.distance_threshold = (
            distance_threshold if distance_threshold is not None else settings.novelty_distance_threshold
        )

    def score(self, embedding: np.ndarray) -> tuple[float, bool]:
        distances = [
            self._cosine_distance(embedding, centroid)
            for centroid in self.centroids.values()
        ]
        min_distance = float(min(distances))
        is_novel = min_distance > self.distance_threshold
        return min_distance, is_novel

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        return float(1 - cosine_similarity)
