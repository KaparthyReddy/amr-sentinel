import numpy as np
from unittest.mock import patch

from app.models.novelty_detector import NoveltyDetector


def _make_detector(centroids: dict, threshold: float = 0.35) -> NoveltyDetector:
    with patch("app.models.novelty_detector.joblib.load", return_value=centroids):
        return NoveltyDetector(distance_threshold=threshold)


def test_identical_vector_scores_zero_distance():
    vector = np.array([1.0, 0.0, 0.0])
    detector = _make_detector({"known_class": vector})

    distance, is_novel = detector.score(vector)

    assert distance < 0.01
    assert is_novel is False


def test_orthogonal_vector_flagged_as_novel():
    known = np.array([1.0, 0.0, 0.0])
    novel_query = np.array([0.0, 1.0, 0.0])
    detector = _make_detector({"known_class": known}, threshold=0.35)

    distance, is_novel = detector.score(novel_query)

    assert distance > 0.9  # cosine distance between orthogonal vectors is 1.0
    assert is_novel is True


def test_uses_closest_centroid_among_multiple():
    close_centroid = np.array([1.0, 0.1, 0.0])
    far_centroid = np.array([0.0, 0.0, 1.0])
    query = np.array([1.0, 0.0, 0.0])

    detector = _make_detector({"close": close_centroid, "far": far_centroid}, threshold=0.35)
    distance, is_novel = detector.score(query)

    assert distance < 0.1
    assert is_novel is False
