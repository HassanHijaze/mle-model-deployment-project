"""Train the trip-duration models and register them in the MLflow server.

Same steps as notebooks/exploration.ipynb:
  1. rf-baseline:       baseline settings, 500k-trip sample, scored on validation
  2. rf-final-all-data: Optuna settings, all Jan 1-20 trips, scored on validation and test

Both models are registered as new versions of "nyc-taxi-duration-rf" with no alias.
Set the "champion" alias yourself in the MLflow UI.

The MLflow server must be running (docker compose up -d); its address comes from
MLFLOW_TRACKING_URI (see src/experiment.py).

Needs a lot of RAM: the all-data model is trained on about 2.1M trips.

Run from the project root:  uv run python -m src.train
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error

from src.experiment import log_model_run, train_and_log
from src.features import build_features, build_zone_xy

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
MODEL_NAME = "nyc-taxi-duration-rf"

BASELINE_PARAMS = {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 5}
# Best settings found with Optuna (on the 500k sample) in notebooks/exploration.ipynb
BEST_PARAMS = {"n_estimators": 200, "max_depth": 25, "min_samples_leaf": 8, "max_features": 0.5}


def load_january() -> pd.DataFrame:
    """All January 2025 trips of 1-60 minutes, sorted by pickup time."""
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]
    df = pd.read_parquet(DATA_URL, columns=cols)

    pickup = df["tpep_pickup_datetime"]
    df = df[(pickup >= "2025-01-01") & (pickup < "2025-02-01")]
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    return df[(df["duration"] >= 1) & (df["duration"] <= 60)]


def load_data(n: int = 500_000) -> pd.DataFrame:
    """Random sample of n January trips, back in time order (same as the notebook)."""
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
                  run_name="rf-baseline", register_name=MODEL_NAME)
    del sample, X_s, y_s

    # 2. Final model on all Jan 1-20 trips, scored on validation and test
    X, y = build_features(full, zone_xy), full["duration"]
    tr, va, te = split_by_time(full)
    model = RandomForestRegressor(**BEST_PARAMS, n_jobs=-1, random_state=42)
    model.fit(X[tr], y[tr])

    val_pred, test_pred = model.predict(X[va]), model.predict(X[te])
    metrics = {
        "val_rmse": root_mean_squared_error(y[va], val_pred),
        "val_rmsle": root_mean_squared_log_error(y[va], val_pred),
        "test_rmse": root_mean_squared_error(y[te], test_pred),
        "test_rmsle": root_mean_squared_log_error(y[te], test_pred),
    }
    log_model_run(model, BEST_PARAMS, metrics, X.columns,
                  run_name="rf-final-all-data", register_name=MODEL_NAME)


if __name__ == "__main__":
    main()