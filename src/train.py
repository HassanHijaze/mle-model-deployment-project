"""Train the trip-duration models and log them to the MLflow server.

Steps (one MLflow run each):
  1. baseline-overall-median     median duration of the training trips    (src/baselines.py)
  2. baseline-zone-pair-median   median per zone pair, unseen pairs get the overall median
  3. rf-baseline                 Random Forest, BASELINE_PARAMS, fitted on the 500k-trip
                                 sample's trips of the training period. The Jan 1-20 fit is
                                 logged with its zone table and REGISTERED (no alias).
     All three are scored the same way (see evaluation_periods):
       validation: all Jan 21-25 trips, statistics/fit from Jan 1-20  -> val_rmse, val_mae
       test:       all Jan 26-31 trips, statistics/fit from Jan 1-25  -> test_rmse, test_mae
     plus the error segments (val_/test_<segment>_rmse/mae).
  4. rf-final-jan1-25:  tuned params (200 trees), trained on all Jan 1-25 trips and scored
                        once on the test days (Jan 26-31). Logs params, dataset periods and
                        test metrics only. NO model artifact: it is several GB and not served.

Tuned params: the best params of the latest finished "optuna-tuning" run in MLflow
(`uv run nyc-taxi tune`, see src/tuning.py). If there is no such run, BEST_PARAMS
below is used instead and a note is printed (see load_tuned_params()).

The model to serve is chosen and registered by `uv run nyc-taxi select`.
Registered models get no alias. Set the "champion" alias yourself in the MLflow UI.

The MLflow server must be running (docker compose up -d postgres mlflow); its address
comes from MLFLOW_TRACKING_URI (see src/experiment.py).

The January data is downloaded once into data/ (only the columns used) and read
from there afterward.

Runtime: roughly 15-30 min, peak RAM about 6-8 GB (the 200-tree model on ~2.7M trips).

Run from the project root:  uv run nyc-taxi train   (or uv run python -m src.train)
"""

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error

from src import serving_selection as ss
from src.baselines import NAIVE_BASELINES, RF_BASELINE_RUN
from src.bundle import period_of
from src.experiment import EXPERIMENT, TRACKING_URI, log_model_run
from src.features import FEATURES, build_features, build_zone_xy

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
DATA_PATH = ROOT / "data" / "yellow_tripdata_2025-01.parquet"
COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]
MODEL_NAME = "nyc-taxi-duration-rf"
BASELINE_RUN_NAME = RF_BASELINE_RUN  # "rf-baseline"
FINAL_RUN_NAME = "rf-final-jan1-25"
TUNING_RUN_NAME = "optuna-tuning"
SAMPLE_SIZE = 500_000

BASELINE_PARAMS = {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 5}
# Fallback only: used when MLflow has no finished "optuna-tuning" run (see load_tuned_params)
BEST_PARAMS = {"n_estimators": 200, "max_depth": 25, "min_samples_leaf": 8, "max_features": 0.5}


def download_january() -> None:
    """Download the January parquet file once, keeping only the columns we use.

    The file is written to a temporary name first and renamed when complete, so an
    interrupted download never leaves a broken file under DATA_PATH.
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_PATH.with_suffix(".parquet.tmp")
    pd.read_parquet(DATA_URL, columns=COLUMNS).to_parquet(tmp_path)
    tmp_path.replace(DATA_PATH)  # atomic rename


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


def load_data(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Random sample of n January trips, back in time order (same sample as the notebook)."""
    return load_january().sample(n=n, random_state=42).sort_index()


def split_by_time(df: pd.DataFrame):
    """Boolean masks: train Jan 1-20, validation Jan 21-25, test Jan 26-31."""
    pickup = df["tpep_pickup_datetime"]
    train = pickup < "2025-01-21"
    val = (pickup >= "2025-01-21") & (pickup < "2025-01-26")
    test = pickup >= "2025-01-26"
    return train, val, test


def _parse_param(value: str):
    """MLflow stores params as strings: '200' -> 200, '0.5' -> 0.5, '1.0' -> 1.0 (a float:
    max_features=1.0 means all features, max_features=1 would mean one feature)."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def load_tuned_params() -> tuple[dict, str]:
    """Best params of the latest finished "optuna-tuning" run, or BEST_PARAMS as a fallback.

    Returns (params, source) and prints which params are used and where they come from.
    """
    try:
        run = ss.latest_run(TUNING_RUN_NAME)
    except LookupError:
        print(
            f"Note: no finished '{TUNING_RUN_NAME}' run in MLflow - using the fallback BEST_PARAMS "
            f"from src/train.py. Run `uv run nyc-taxi tune` to tune them."
        )
        print("Params used:", BEST_PARAMS)
        return dict(BEST_PARAMS), "BEST_PARAMS fallback (src/train.py)"

    params = {
        k.removeprefix("best_"): _parse_param(v) for k, v in run.data.params.items() if k.startswith("best_")
    }
    if set(params) != set(BEST_PARAMS):
        print(
            f"Note: run {run.info.run_id} ({TUNING_RUN_NAME}) has unexpected best params "
            f"{sorted(params)} - using the fallback BEST_PARAMS from src/train.py."
        )
        print("Params used:", BEST_PARAMS)
        return dict(BEST_PARAMS), "BEST_PARAMS fallback (src/train.py)"
    source = f"{TUNING_RUN_NAME} run {run.info.run_id}"
    print(f"Params used (from {source}):", params)
    return params, source


def evaluation_periods(full: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """{"val": (Jan 1-20 trips, Jan 21-25 trips), "test": (Jan 1-25 trips, Jan 26-31 trips)}.

    The first frame of each pair is the only data a baseline may learn from.
    """
    tr, va, te = split_by_time(full)
    return {"val": (full[tr], full[va]), "test": (full[tr | va], full[te])}


def score_predictions(split: str, train_df: pd.DataFrame, eval_df: pd.DataFrame, pred) -> dict[str, float]:
    """<split>_rmse/mae and <split>_<segment>_rmse/mae (segments use training-period pair counts)."""
    masks = ss.segment_masks(eval_df, ss.zone_pair_counts(train_df))
    metrics = ss.regression_metrics(eval_df["duration"], pred, prefix=f"{split}_")
    metrics.update({f"{split}_{k}": v for k, v in ss.segment_metrics(eval_df, pred, masks).items()})
    return metrics


def _period_params(periods: dict) -> dict[str, str | int]:
    """Training and evaluation periods of both splits, as MLflow params."""
    params: dict[str, str | int] = {}
    for split, (train_df, eval_df) in periods.items():
        for role, df in (("train", train_df), ("eval", eval_df)):
            pickup = df["tpep_pickup_datetime"]
            params[f"{split}_{role}_period"] = f"{pickup.min():%Y-%m-%d}..{pickup.max():%Y-%m-%d}"
            params[f"{split}_{role}_rows"] = len(df)
    return params


def log_naive_baselines(full: pd.DataFrame) -> dict[str, dict[str, float]]:
    """One MLflow run per naive baseline (src/baselines.py), scored on validation and test.

    Returns {run name: metrics}.
    """
    periods = evaluation_periods(full)
    results = {}
    for run_name, (predict, description) in NAIVE_BASELINES.items():
        metrics = {}
        for split, (train_df, eval_df) in periods.items():
            metrics.update(score_predictions(split, train_df, eval_df, predict(train_df, eval_df)))
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"baseline": description, **_period_params(periods)})
            mlflow.log_metrics(metrics)
        print(run_name, {k: round(metrics[k], 4) for k in ("val_rmse", "val_mae", "test_rmse", "test_mae")})
        results[run_name] = metrics
    return results


def train_baseline(full: pd.DataFrame, X: pd.DataFrame, zone_xy: pd.DataFrame) -> str:
    """rf-baseline: BASELINE_PARAMS fitted on the 500k sample's trips of each training period.

    Validation: fit on the sample's Jan 1-20 trips, scored on all Jan 21-25 trips.
    Test: refit on the sample's Jan 1-25 trips, scored on all Jan 26-31 trips.
    The Jan 1-20 fit is logged with its zone table and registered (no alias), as before.
    Returns the registered model version.
    """
    sample_index = full.sample(n=SAMPLE_SIZE, random_state=42).index  # same sample as tuning
    metrics, logged_model, logged_index = {}, None, None
    for split, (train_df, eval_df) in evaluation_periods(full).items():
        fit_index = train_df.index[train_df.index.isin(sample_index)]
        model = ss.fit_rf(BASELINE_PARAMS, X.loc[fit_index], full.loc[fit_index, "duration"])
        pred = model.predict(X.loc[eval_df.index])
        metrics.update(score_predictions(split, train_df, eval_df, pred))
        if split == "val":
            logged_model, logged_index = model, fit_index
        del model

    params = {
        **BASELINE_PARAMS,
        "sample_size": SAMPLE_SIZE,
        "logged_model": "fit on the sample's Jan 1-20 trips",
        "test_scored_with": "refit on the sample's Jan 1-25 trips",
    }
    version = log_model_run(
        logged_model,
        params,
        metrics,
        X.columns,
        BASELINE_RUN_NAME,
        register_name=MODEL_NAME,
        zone_xy=zone_xy,
        X=X.loc[logged_index],
        training_period=period_of(full.loc[logged_index]),
    )
    return version


def train_final(
    full: pd.DataFrame, X: pd.DataFrame, params: dict, params_source: str = ""
) -> dict[str, float]:
    """rf-final-jan1-25: the tuned params on Jan 1-25, scored once on Jan 26-31.

    Logs params, dataset periods and test metrics only (no model artifact):
    test_rmse/mae/rmsle, test_<segment>_rmse/mae and test_<segment>_trips.
    The baselines it is compared with are their own runs (log_naive_baselines, train_baseline).
    Returns the logged metrics.
    """
    y = full["duration"]
    tr, va, te = split_by_time(full)
    dev = tr | va
    dev_df, test_df = full[dev], full[te]

    model = ss.fit_rf(params, X[dev], y[dev])
    pred = model.predict(X[te])
    del model  # several GB; not logged

    metrics = score_predictions("test", dev_df, test_df, pred)
    metrics["test_rmsle"] = float(root_mean_squared_log_error(y[te], pred))
    masks = ss.segment_masks(test_df, ss.zone_pair_counts(dev_df))
    metrics.update({f"test_{k}_trips": float(n) for k, n in ss.segment_sizes(masks).items()})

    with mlflow.start_run(run_name=FINAL_RUN_NAME):
        mlflow.log_params(params)
        mlflow.log_params({"features": ",".join(FEATURES), "params_source": params_source})
        mlflow.log_metrics(metrics)
        ss.log_datasets({"training": dev_df, "test": test_df}, source=DATA_URL)

    print(FINAL_RUN_NAME, {k: round(metrics[k], 4) for k in ("test_rmse", "test_mae")})
    return metrics


def main() -> dict:
    """Log the baselines and the final test score.

    Returns {"baseline_version": registered version of rf-baseline}.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)  # fails fast if the MLflow server is not running

    params, source = load_tuned_params()
    full = load_january()
    zone_xy = build_zone_xy(ROOT / "data")
    X = build_features(full, zone_xy)  # built once, shared by every model

    log_naive_baselines(full)
    baseline_version = train_baseline(full, X, zone_xy)
    train_final(full, X, params, source)
    return {"baseline_version": baseline_version}


if __name__ == "__main__":
    main()
