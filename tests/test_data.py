"""Unit tests for the data cleaning and time split, using small made-up trips (no download)."""

import pandas as pd

from src.train import clean_trips, split_by_time


def make_trips(rows):
    """rows: list of (pickup time, dropoff time) as strings."""
    df = pd.DataFrame(rows, columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df = df.apply(lambda col: pd.to_datetime(col, format="ISO8601"))
    df["PULocationID"], df["DOLocationID"] = 161, 236
    return df


def test_clean_trips_keeps_only_1_to_60_minutes():
    trips = make_trips(
        [
            ("2025-01-10 08:00", "2025-01-10 08:00:30"),  # 0.5 min  -> removed
            ("2025-01-10 08:00", "2025-01-10 08:01"),  # 1 min    -> kept
            ("2025-01-10 08:00", "2025-01-10 08:30"),  # 30 min   -> kept
            ("2025-01-10 08:00", "2025-01-10 09:00"),  # 60 min   -> kept
            ("2025-01-10 08:00", "2025-01-10 09:01"),  # 61 min   -> removed
            ("2025-01-10 08:00", "2025-01-10 07:50"),  # negative -> removed
        ]
    )
    cleaned = clean_trips(trips)
    assert sorted(cleaned["duration"].tolist()) == [1, 30, 60]


def test_clean_trips_keeps_only_january_and_sorts():
    trips = make_trips(
        [
            ("2025-01-20 10:00", "2025-01-20 10:10"),
            ("2024-12-31 23:00", "2024-12-31 23:10"),  # December -> removed
            ("2025-01-05 10:00", "2025-01-05 10:10"),
            ("2025-02-01 00:05", "2025-02-01 00:15"),  # February -> removed
        ]
    )
    cleaned = clean_trips(trips)
    assert cleaned["tpep_pickup_datetime"].dt.month.eq(1).all()
    assert cleaned["tpep_pickup_datetime"].is_monotonic_increasing
    assert len(cleaned) == 2


def test_split_by_time_assigns_each_day_correctly():
    trips = make_trips(
        [
            ("2025-01-01 00:00", "2025-01-01 00:10"),  # train
            ("2025-01-20 23:59", "2025-01-21 00:09"),  # train (last minute of Jan 20)
            ("2025-01-21 00:00", "2025-01-21 00:10"),  # validation
            ("2025-01-25 23:59", "2025-01-26 00:09"),  # validation
            ("2025-01-26 00:00", "2025-01-26 00:10"),  # test
            ("2025-01-31 23:00", "2025-01-31 23:10"),  # test
        ]
    )
    train, val, test = split_by_time(trips)
    assert train.tolist() == [True, True, False, False, False, False]
    assert val.tolist() == [False, False, True, True, False, False]
    assert test.tolist() == [False, False, False, False, True, True]


def test_split_by_time_has_no_overlap_and_covers_all():
    trips = make_trips([(f"2025-01-{d:02d} 12:00", f"2025-01-{d:02d} 12:10") for d in range(1, 32)])
    train, val, test = split_by_time(trips)
    assert not (train & val).any() and not (val & test).any() and not (train & test).any()
    assert (train | val | test).all()
