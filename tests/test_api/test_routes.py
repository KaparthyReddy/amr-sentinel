from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: MagicMock()

    app.state.embedding_service = MagicMock()
    app.state.embedding_service.model_name = "esm2_t6_8M_UR50D"
    app.state.embedding_service.embed.return_value = np.zeros(320)

    app.state.classifier_service = MagicMock()
    app.state.classifier_service.known_mechanisms = ["antibiotic efflux", "antibiotic inactivation"]
    app.state.classifier_service.predict.return_value = ("antibiotic efflux", 0.85)

    app.state.novelty_detector = MagicMock()
    app.state.novelty_detector.score.return_value = (0.12, False)

    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_analyze_valid_sequence(client):
    response = client.post("/analyze", json={"sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ"})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_mechanism"] == "antibiotic efflux"
    assert body["is_novel"] is False


def test_analyze_rejects_invalid_amino_acids(client):
    response = client.post("/analyze", json={"sequence": "MKTAYIAKQ123INVALID"})
    assert response.status_code == 422


def test_analyze_rejects_too_short_sequence(client):
    response = client.post("/analyze", json={"sequence": "MKT"})
    assert response.status_code == 422
