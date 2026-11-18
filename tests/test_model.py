"""Model tests: train-serving skew, sanity checks and behavioral tests.

Zones used: 161 Midtown Center, 162 Midtown East, 236 Upper East Side North,
74 East Harlem North, 132 JFK Airport.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import root_mean_squared_error

from src.features import build_features
from src.train import split_by_time


def predict(model, zone_xy, pu, do, when):
    """Prediction for one trip, built with the same feature code as the API."""
    trips = pd.DataFrame([{"PULocationID": pu, "DOLocationID": do,
                           "tpep_pickup_datetime": pd.Timestamp(when)}])
    return float(model.predict(build_features(trips, zone_xy))[0])


# ---------- Train-serving skew ----------

@pytest.mark.parametrize("pu, do, when", [
    (161, 236, "2025-01-27 08:30"),
    (161, 132, "2025-01-28 17:00"),
    (132, 161, "2025-01-29 03:00"),
    (74, 162, "2025-01-01 12:00"),   # holiday
    (161, 264, "2025-01-30 10:00"),  # unknown zone
])
def test_api_matches_model_predictions(client, model, zone_xy, pu, do, when):
    r = client.post("/predict", json={"PULocationID": pu, "DOLocationID": do,
                                      "pickup_datetime": pd.Timestamp(when).isoformat()})
    assert r.status_code == 200
    assert r.json()["duration_minutes"] == pytest.approx(predict(model, zone_xy, pu, do, when), abs=0.01)


# ---------- Sanity checks ----------

def test_predictions_in_valid_range(model, zone_xy):
    rng = np.random.default_rng(42)
    trips = pd.DataFrame({
        "PULocationID": rng.integers(1, 266, 500),
        "DOLocationID": rng.integers(1, 266, 500),
        "tpep_pickup_datetime": pd.Timestamp("2025-01-27") + pd.to_timedelta(rng.integers(0, 24 * 60, 500), unit="min"),
    })
    pred = model.predict(build_features(trips, zone_xy))
    assert not np.isnan(pred).any()
    assert (pred >= 1).all() and (pred <= 60).all()   # trained on 1-60 minute trips


def test_model_beats_mean_baseline(model, zone_xy, data):
    train, _, test = split_by_time(data)
    X_test, y_test = build_features(data[test], zone_xy), data.loc[test, "duration"]
    model_rmse = root_mean_squared_error(y_test, model.predict(X_test))
    mean_rmse = root_mean_squared_error(y_test, np.full(len(y_test), data.loc[train, "duration"].mean()))
    assert model_rmse < mean_rmse


# ---------- Behavioral tests ----------

def test_longer_trip_predicts_longer_duration(model, zone_xy):
    near = predict(model, zone_xy, 161, 162, "2025-01-28 11:00")  # Midtown -> Midtown East
    far = predict(model, zone_xy, 161, 74, "2025-01-28 11:00")    # Midtown -> East Harlem
    assert far > near


def test_rush_hour_slower_than_night(model, zone_xy):
    rush = predict(model, zone_xy, 161, 236, "2025-01-28 17:00")
    night = predict(model, zone_xy, 161, 236, "2025-01-28 03:00")
    assert rush > night


def test_airport_trip_longer_than_local(model, zone_xy):
    airport = predict(model, zone_xy, 161, 132, "2025-01-28 11:00")  # Midtown -> JFK
    local = predict(model, zone_xy, 161, 162, "2025-01-28 11:00")    # Midtown -> Midtown East
    assert airport > local + 10


def test_minute_change_does_not_change_much(model, zone_xy):
    a = predict(model, zone_xy, 161, 236, "2025-01-28 08:30")
    b = predict(model, zone_xy, 161, 236, "2025-01-28 08:31")
    assert abs(a - b) < 1
