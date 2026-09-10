from unittest.mock import MagicMock, patch
from datetime import datetime

from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db
from app.db.models import SurveillanceLog


def _make_log(mechanism: str, is_novel: bool) -> SurveillanceLog:
    log = SurveillanceLog(
        sequence_hash="abc123",
        sequence_length=200,
        predicted_mechanism=mechanism,
        confidence=0.9,
        novelty_distance=0.1,
        is_novel=is_novel,
    )
    log.created_at = datetime.utcnow()
    return log


def test_summary_empty_window():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = []
    app.dependency_overrides[get_db] = lambda: mock_db

    client = TestClient(app)
    response = client.get("/surveillance/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["total_analyzed"] == 0
    assert body["novel_rate"] == 0.0

    app.dependency_overrides.clear()


def test_summary_aggregates_by_mechanism_and_novelty():
    logs = [
        _make_log("antibiotic inactivation", is_novel=False),
        _make_log("antibiotic inactivation", is_novel=False),
        _make_log("antibiotic efflux", is_novel=True),
    ]
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = logs
    app.dependency_overrides[get_db] = lambda: mock_db

    client = TestClient(app)
    response = client.get("/surveillance/summary?window_minutes=1440")

    assert response.status_code == 200
    body = response.json()
    assert body["total_analyzed"] == 3
    assert body["by_mechanism"]["antibiotic inactivation"] == 2
    assert body["by_mechanism"]["antibiotic efflux"] == 1
    assert body["novel_count"] == 1
    assert abs(body["novel_rate"] - 0.3333) < 0.001

    app.dependency_overrides.clear()
