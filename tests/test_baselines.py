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
from src.train import evaluation_periods, score_predictions


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


def test_evaluation_periods_learn_from_training_days_only():
    days = pd.date_range("2025-01-01 08:00", "2025-01-31 08:00", freq="D")
    full = pd.DataFrame(
        {"tpep_pickup_datetime": days, "PULocationID": 1, "DOLocationID": 2, "duration": 10.0}
    )
    periods = evaluation_periods(full)

    val_train, val_eval = periods["val"]
    assert val_train["tpep_pickup_datetime"].max() < pd.Timestamp("2025-01-21")
    assert val_eval["tpep_pickup_datetime"].dt.day.tolist() == [21, 22, 23, 24, 25]

    test_train, test_eval = periods["test"]
    assert test_train["tpep_pickup_datetime"].max() < pd.Timestamp("2025-01-26")
    assert test_train["tpep_pickup_datetime"].dt.day.max() == 25
    assert test_eval["tpep_pickup_datetime"].dt.day.min() == 26


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
