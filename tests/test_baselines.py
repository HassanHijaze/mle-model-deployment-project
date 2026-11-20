"""Unit tests for the baselines, on small made-up trips (no download, no MLflow)."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    NAIVE_BASELINES,
    missing_runs_message,
    overall_median_predict,
    zone_pair_median_predict,
)
from src.evaluation import score_predictions
from src.train import make_splits, training_sample


def trips(rows, start=0):
    """rows: list of (PULocationID, DOLocationID, duration). Index starts at `start`."""
    df = pd.DataFrame(rows, columns=["PULocationID", "DOLocationID", "duration"])
    df.index = range(start, start + len(df))
    return df


TRAIN = trips(
    [
        (1, 2, 10.0),
        (1, 2, 20.0),
        (1, 2, 12.0),  # pair (1, 2): median 12
        (3, 4, 30.0),
        (3, 4, 40.0),  # pair (3, 4): median 35
    ]
)  # overall training median: 20


def test_overall_median_uses_training_durations_only():
    eval_df = trips([(1, 2, 999.0), (5, 6, -50.0), (3, 4, 0.0)], start=100)
    np.testing.assert_array_equal(overall_median_predict(TRAIN, eval_df), [20.0, 20.0, 20.0])


def test_zone_pair_median_per_training_pair():
    eval_df = trips([(3, 4, 0.0), (1, 2, 0.0)], start=100)
    np.testing.assert_array_equal(zone_pair_median_predict(TRAIN, eval_df), [35.0, 12.0])


def test_zone_pair_median_unseen_pair_falls_back_to_overall_training_median():
    eval_df = trips([(1, 2, 5.0), (2, 1, 5.0), (9, 9, 5.0)], start=100)  # (2, 1) is not (1, 2)
    np.testing.assert_array_equal(zone_pair_median_predict(TRAIN, eval_df), [12.0, 20.0, 20.0])


def test_zone_pair_median_ignores_evaluation_durations():
    eval_a = trips([(1, 2, 1.0), (3, 4, 1.0)], start=100)
    eval_b = trips([(1, 2, 500.0), (3, 4, 500.0)], start=100)
    np.testing.assert_array_equal(
        zone_pair_median_predict(TRAIN, eval_a), zone_pair_median_predict(TRAIN, eval_b)
    )


@pytest.mark.parametrize("predict", [overall_median_predict, zone_pair_median_predict])
def test_one_prediction_per_evaluation_trip_in_order(predict):
    eval_df = trips([(3, 4, 0.0), (7, 8, 0.0), (1, 2, 0.0), (3, 4, 0.0)], start=500)
    pred = predict(TRAIN, eval_df)
    assert pred.shape == (len(eval_df),)
    if predict is zone_pair_median_predict:
        np.testing.assert_array_equal(pred, [35.0, 20.0, 12.0, 35.0])


def test_every_naive_baseline_has_a_function_and_a_description():
    for run_name, (predict, description) in NAIVE_BASELINES.items():
        assert run_name.startswith("baseline-") and callable(predict) and description


def january(hours_step: int = 6) -> pd.DataFrame:
    pickup = pd.date_range("2025-01-01 00:00", "2025-01-31 23:00", freq=f"{hours_step}h")
    return pd.DataFrame(
        {"tpep_pickup_datetime": pickup, "PULocationID": 1, "DOLocationID": 2, "duration": 10.0}
    )


def test_splits_are_fixed_january_periods():
    splits = make_splits(january())
    assert splits.train["tpep_pickup_datetime"].max() < pd.Timestamp("2025-01-21")
    assert splits.val["tpep_pickup_datetime"].dt.day.unique().tolist() == [21, 22, 23, 24, 25]
    assert splits.test["tpep_pickup_datetime"].dt.day.min() == 26
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(january())


def test_training_sample_uses_training_days_only_and_is_repeatable():
    splits = make_splits(january(hours_step=1))
    sample = training_sample(splits.train, n=50)
    assert len(sample) == 50
    assert sample.index.isin(splits.train.index).all()
    assert sample["tpep_pickup_datetime"].is_monotonic_increasing
    assert sample.index.equals(training_sample(splits.train, n=50).index)  # same seed, same trips
    assert training_sample(splits.train, n=10_000).index.equals(splits.train.index)  # smaller than n: all


def test_score_predictions_logs_split_prefixed_metrics():
    eval_df = trips([(1, 2, 10.0), (3, 4, 20.0)], start=100)
    metrics = score_predictions("val", TRAIN, eval_df, np.array([12.0, 20.0]))
    assert metrics["val_rmse"] == pytest.approx(np.sqrt(2.0))
    assert metrics["val_mae"] == pytest.approx(1.0)
    assert {"val_short_rmse", "val_airport_mae", "val_uncommon_pair_rmse"} <= set(metrics)


def test_missing_runs_message():
    assert missing_runs_message([]) == ""
    message = missing_runs_message(["baseline-overall-median"])
    assert "baseline-overall-median" in message and "uv run nyc-taxi all" in message
