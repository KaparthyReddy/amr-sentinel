import numpy as np
from unittest.mock import MagicMock, patch

from app.models.classifier_service import ClassifierService


def test_predict_returns_label_and_confidence():
    mock_classifier = MagicMock()
    mock_classifier.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])

    mock_label_encoder = MagicMock()
    mock_label_encoder.inverse_transform.return_value = ["antibiotic efflux"]
    mock_label_encoder.classes_ = ["a", "antibiotic efflux", "c"]

    with patch("app.models.classifier_service.joblib.load", side_effect=[mock_classifier, mock_label_encoder]):
        service = ClassifierService()

    label, confidence = service.predict(np.zeros(320))

    assert label == "antibiotic efflux"
    assert confidence == 0.7


def test_known_mechanisms_property():
    mock_classifier = MagicMock()
    mock_label_encoder = MagicMock()
    mock_label_encoder.classes_ = ["efflux", "inactivation"]

    with patch("app.models.classifier_service.joblib.load", side_effect=[mock_classifier, mock_label_encoder]):
        service = ClassifierService()

    assert service.known_mechanisms == ["efflux", "inactivation"]
