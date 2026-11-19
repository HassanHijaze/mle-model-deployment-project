"""Integration tests with the real champion model: model loading and API endpoints.

Fixtures (model, client, ...) live in tests/conftest.py. These tests skip themselves
when no MLflow server or champion model is available (e.g. in CI).
Data cleaning and the time split are tested on made-up data in tests/test_data.py.
"""

import pandas as pd
import pytest

from src.features import build_features

TRIP = {"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}


# ---------- Champion model ----------

def test_champion_model_loads(model, zone_xy):
    trips = pd.DataFrame([{"PULocationID": 161, "DOLocationID": 236,
                           "tpep_pickup_datetime": pd.Timestamp("2025-01-27 08:30")}])
    pred = model.predict(build_features(trips, zone_xy))
    assert len(pred) == 1
    assert 0 < pred[0] < 120


# ---------- API ----------

def test_predict_endpoint_returns_duration(client, champion_version):
    r = client.post("/predict", json=TRIP)
    assert r.status_code == 200
    body = r.json()
    assert body["duration_minutes"] > 0
    assert body["model_version"] == champion_version


@pytest.mark.parametrize("bad_trip", [
    {**TRIP, "PULocationID": 0},
    {**TRIP, "DOLocationID": 300},
    {k: v for k, v in TRIP.items() if k != "pickup_datetime"},
    {**TRIP, "pickup_datetime": "2025-01-27T08:30:00Z"},
    {**TRIP, "pickup_datetime": "2025-01-27T08:30:00+00:00"},
])
def test_predict_rejects_invalid_input(client, bad_trip):
    r = client.post("/predict", json=bad_trip)
    assert r.status_code == 422


def test_reload_endpoint_returns_versions(client, champion_version):
    r = client.post("/reload")
    assert r.status_code == 200
    assert r.json()["current_version"] == champion_version