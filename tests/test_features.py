"""Unit tests for src/features.py, using small made-up data (no downloads)."""

import pandas as pd
import pytest

from src.features import FEATURES, build_features

# Fake zone centers: zone 2 is 3 miles east of zone 1, zone 3 is 4 miles north of zone 1
FEET_PER_MILE = 5280
ZONE_XY = pd.DataFrame(
    {
        "x": [0, 3 * FEET_PER_MILE, 0],
        "y": [0, 0, 4 * FEET_PER_MILE],
        "lat": [40.70, 40.70, 40.76],
        "lon": [-74.00, -73.94, -74.00],
    },
    index=pd.Index([1, 2, 3], name="LocationID"),
)


def make_trips(rows):
    """rows: list of (PULocationID, DOLocationID, pickup time as string)."""
    return pd.DataFrame(
        rows, columns=["PULocationID", "DOLocationID", "tpep_pickup_datetime"]
    ).assign(tpep_pickup_datetime=lambda d: pd.to_datetime(d["tpep_pickup_datetime"]))


def test_build_features_returns_expected_columns():
    X = build_features(make_trips([(1, 2, "2025-01-27 08:30")]), ZONE_XY)
    assert list(X.columns) == FEATURES


def test_hour_and_day_of_week():
    X = build_features(make_trips([(1, 2, "2025-01-27 08:30")]), ZONE_XY)  # a Monday
    assert X.loc[0, "hour"] == 8
    assert X.loc[0, "day_of_week"] == 0


def test_is_holiday_flags_holidays():
    trips = make_trips([
        (1, 2, "2025-01-01 12:00"),  # New Year's Day
        (1, 2, "2025-01-20 12:00"),  # MLK Day
        (1, 2, "2025-01-15 12:00"),  # normal Wednesday
    ])
    X = build_features(trips, ZONE_XY)
    assert X["is_holiday"].tolist() == [1, 1, 0]


def test_straight_dist_same_zone_is_zero():
    X = build_features(make_trips([(1, 1, "2025-01-27 08:30")]), ZONE_XY)
    assert X.loc[0, "straight_dist"] == 0


def test_straight_dist_is_symmetric():
    X = build_features(make_trips([
        (1, 2, "2025-01-27 08:30"),
        (2, 1, "2025-01-27 08:30"),
    ]), ZONE_XY)
    assert X.loc[0, "straight_dist"] == pytest.approx(3.0)
    assert X.loc[1, "straight_dist"] == pytest.approx(3.0)


def test_unknown_zone_gets_minus_one():
    X = build_features(make_trips([(1, 264, "2025-01-27 08:30")]), ZONE_XY)
    assert X.loc[0, "DO_lat"] == -1
    assert X.loc[0, "DO_lon"] == -1
    assert X.loc[0, "straight_dist"] == -1
    assert not X.isna().any().any()
