"""Reference baselines and the MLflow runs that hold their scores.

Every baseline learns its statistics from the training trips it is given and nothing else:

    overall_median_predict      the median duration of the training trips, for every trip
    zone_pair_median_predict    the median duration of the training trips with the same
                                pickup/drop-off zone pair; unseen pairs get the overall median

`nyc-taxi train` (src.train) scores them on validation (statistics from Jan 1-20)
and test (statistics from Jan 1-25) and logs one MLflow run per baseline, next to the
Random Forest baseline "rf-baseline". The notebooks only read these runs
(read_baseline_runs); they never compute a baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OVERALL_MEDIAN_RUN = "baseline-overall-median"
ZONE_PAIR_MEDIAN_RUN = "baseline-zone-pair-median"
RF_BASELINE_RUN = "rf-baseline"

# Display label -> MLflow run name, in the order the notebooks show them
BASELINE_RUNS = {
    "Overall median": OVERALL_MEDIAN_RUN,
    "Zone-pair median": ZONE_PAIR_MEDIAN_RUN,
    "RF baseline": RF_BASELINE_RUN,
}
MISSING_RUN_HINT = "Run `uv run nyc-taxi all` first (MLflow server must be running)."

ZONE_PAIR = ["PULocationID", "DOLocationID"]


def overall_median_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    """Predict the median training duration for every trip of eval_df."""
    return np.full(len(eval_df), float(train_df["duration"].median()))


def zone_pair_median_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    """Predict the median training duration of each trip's zone pair.

    Zone pairs that do not appear in train_df get the overall training median.
    Only train_df is used to compute the statistics (no leakage from eval_df).
    """
    medians = train_df.groupby(ZONE_PAIR)["duration"].median()
    keys = pd.MultiIndex.from_frame(eval_df[ZONE_PAIR])
    return medians.reindex(keys).fillna(train_df["duration"].median()).to_numpy()


# MLflow run name -> (prediction function, description logged as a param)
NAIVE_BASELINES = {
    OVERALL_MEDIAN_RUN: (overall_median_predict, "median duration of the training trips"),
    ZONE_PAIR_MEDIAN_RUN: (
        zone_pair_median_predict,
        "median duration per pickup/drop-off zone pair of the training trips; "
        "unseen pairs get the overall training median",
    ),
}


def read_baseline_runs(experiment: str = "nyc-taxi-duration") -> tuple[dict, list[str]]:
    """Latest finished run of each baseline: ({label: mlflow Run}, [missing run names])."""
    from src.serving_selection import latest_run

    runs, missing = {}, []
    for label, run_name in BASELINE_RUNS.items():
        try:
            runs[label] = latest_run(run_name, experiment=experiment)
        except LookupError:
            missing.append(run_name)
    return runs, missing


def baseline_metrics_table(runs: dict) -> pd.DataFrame:
    """One row per baseline (label), one column per logged metric."""
    return pd.DataFrame({label: run.data.metrics for label, run in runs.items()}).T


def missing_runs_message(missing: list[str]) -> str:
    """Short message for the notebooks when baseline runs are missing ('' if none)."""
    if not missing:
        return ""
    return f"No finished MLflow run for: {', '.join(missing)}. {MISSING_RUN_HINT}"
