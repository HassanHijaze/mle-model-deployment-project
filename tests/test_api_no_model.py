"""API tests that need no MLflow server and no model, so they also run in CI.

The model and zone map loaders are replaced with "nothing loaded", which lets us
test input validation (422) and the "no model" path (503).
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main

VALID_TRIP = {"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}


@pytest.fixture
def client_no_model(monkeypatch):
    monkeypatch.setattr(main, "load_champion", lambda: (None, None, None))
    with TestClient(main.app) as c:
        yield c


@pytest.mark.parametrize("bad_trip", [
    {**VALID_TRIP, "PULocationID": 0},
    {**VALID_TRIP, "DOLocationID": 300},
    {k: v for k, v in VALID_TRIP.items() if k != "pickup_datetime"},
    {**VALID_TRIP, "pickup_datetime": "2025-01-27T08:30:00Z"},
    {**VALID_TRIP, "pickup_datetime": "2025-01-27T08:30:00+00:00"},
])
def test_predict_rejects_invalid_input(client_no_model, bad_trip):
    assert client_no_model.post("/predict", json=bad_trip).status_code == 422


def test_predict_returns_503_without_model(client_no_model):
    r = client_no_model.post("/predict", json=VALID_TRIP)
    assert r.status_code == 503
    assert "No model loaded" in r.json()["detail"]


def test_health_reports_nothing_loaded(client_no_model):
    body = client_no_model.get("/health").json()
    assert body["status"] == "running"
    assert body["model_loaded"] is False
    assert body["zones_loaded"] is False