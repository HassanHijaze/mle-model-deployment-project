"""Train the trip-duration models and register them in the MLflow server.

Steps:
  1. rf-baseline:        baseline settings, 500k-trip sample, scored on validation (registered)
  2. rf-final-jan1-25:   BEST_PARAMS, trained on all Jan 1-25 trips, tested once (registered)

This script does NOT run hyperparameter tuning. BEST_PARAMS are copied from the
tune_random_forest() run (Optuna, 20 trials on the 500k sample) in
notebooks/exploration.ipynb. If you re-run the tuning there and get different
settings, update BEST_PARAMS here by hand.

Each model is logged with its zone table (zone_xy.csv), which the API loads.
Registered models get no alias. Set the "champion" alias yourself in the MLflow UI.

The MLflow server must be running (docker compose up -d); its address comes from
MLFLOW_TRACKING_URI (see src/experiment.py).

The January data is downloaded once into data/ (only the columns used) and read
from there afterward.

Needs a lot of RAM and disk: the final model is trained on about 2.7M trips and can be
several GB once saved. Loading it (API container, tests) also needs several GB of memory.

Run from the project root:  uv run python -m src.train
"""

from pathlib import Path

import pandas as pd

from src.experiment import train_and_log
from src.features import build_features, build_zone_xy

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
DATA_PATH = ROOT / "data" / "yellow_tripdata_2025-01.parquet"
COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]
MODEL_NAME = "nyc-taxi-duration-rf"

BASELINE_PARAMS = {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 5}
# Copied from tune_random_forest() in notebooks/exploration.ipynb (not tuned in this script)
BEST_PARAMS = {"n_estimators": 200, "max_depth": 25, "min_samples_leaf": 8, "max_features": 0.5}


def download_january() -> None:
    """Download the January parquet file once, keeping only the columns we use.

    The file is written to a temporary name first and renamed when complete, so an
    interrupted download never leaves a broken file under DATA_PATH.
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_PATH.with_suffix(".parquet.tmp")
    pd.read_parquet(DATA_URL, columns=COLUMNS).to_parquet(tmp_path)
    tmp_path.replace(DATA_PATH)   # atomic rename


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Keep January 2025 pickups, sort by pickup time, add duration (minutes), keep 1-60 minute trips."""
    pickup = df["tpep_pickup_datetime"]
    df = df[(pickup >= "2025-01-01") & (pickup < "2025-02-01")]
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    return df[(df["duration"] >= 1) & (df["duration"] <= 60)]


def load_january() -> pd.DataFrame:
    """All January 2025 trips of 1-60 minutes, sorted by pickup time."""
    if not DATA_PATH.exists():
        download_january()
    return clean_trips(pd.read_parquet(DATA_PATH, columns=COLUMNS))


def load_data(n: int = 500_000) -> pd.DataFrame:
    """Random sample of n January trips, back in time order (same sample as the notebook)."""
    return load_january().sample(n=n, random_state=42).sort_index()


def split_by_time(df: pd.DataFrame):
    """Boolean masks: train Jan 1-20, validation Jan 21-25, test Jan 26-31."""
    pickup = df["tpep_pickup_datetime"]
    train = pickup < "2025-01-21"
    val = (pickup >= "2025-01-21") & (pickup < "2025-01-26")
    test = pickup >= "2025-01-26"
    return train, val, test


def main() -> None:
    full = load_january()
    zone_xy = build_zone_xy(ROOT / "data")

    # 1. Baseline on the 500k sample, scored on validation
    sample = full.sample(n=500_000, random_state=42).sort_index()
    X_s, y_s = build_features(sample, zone_xy), sample["duration"]
    tr_s, va_s, _ = split_by_time(sample)
    train_and_log(BASELINE_PARAMS, X_s[tr_s], y_s[tr_s], X_s[va_s], y_s[va_s],
                  run_name="rf-baseline", register_name=MODEL_NAME, zone_xy=zone_xy)
    del sample, X_s, y_s

    # 2. Final model: train on Jan 1-25 (train + validation), test once on Jan 26-31, register
    X, y = build_features(full, zone_xy), full["duration"]
    tr, va, te = split_by_time(full)
    train_and_log(BEST_PARAMS, X[tr | va], y[tr | va], X[te], y[te],
                  run_name="rf-final-jan1-25", eval_name="test", register_name=MODEL_NAME,
                  zone_xy=zone_xy)


if __name__ == "__main__":
    main()