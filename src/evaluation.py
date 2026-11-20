"""Shared evaluation helpers: metrics, error segments, model fitting and model size.

Used by every pipeline step (src.train, src.tuning, src.selection), so all models are
fitted and scored the same way.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42  # every Random Forest uses this seed: same params + same data = same model
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]  # same skops settings as src.experiment

AIRPORT_ZONES = (1, 132, 138)  # Newark (EWR), JFK, LaGuardia
SHORT_TRIP_MAX_MIN = 5.0  # actual duration < 5 min
LONG_TRIP_MIN_MIN = 30.0  # actual duration > 30 min
UNCOMMON_PAIR_MAX_TRIPS = 10  # zone pair seen fewer than 10 times in the training period
SEGMENTS = ("short", "long", "airport", "uncommon_pair")  # keys of segment_masks()


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """RMSE and MAE in minutes."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}mae": float(np.mean(np.abs(err))),
    }


def zone_pair_counts(train_df: pd.DataFrame) -> pd.Series:
    """Number of training trips per (PULocationID, DOLocationID) pair."""
    return train_df.groupby(["PULocationID", "DOLocationID"]).size()


def segment_masks(eval_df: pd.DataFrame, pair_counts: pd.Series) -> dict[str, np.ndarray]:
    """Boolean masks for the error segments.

    short: actual duration < 5 min; long: actual duration > 30 min;
    airport: pickup or drop-off at EWR/JFK/LGA;
    uncommon_pair: zone pair with fewer than 10 trips in the training period (incl. unseen).
    """
    keys = pd.MultiIndex.from_frame(eval_df[["PULocationID", "DOLocationID"]])
    seen = pair_counts.reindex(keys).fillna(0).to_numpy()
    duration = eval_df["duration"].to_numpy()
    airport = eval_df["PULocationID"].isin(AIRPORT_ZONES) | eval_df["DOLocationID"].isin(AIRPORT_ZONES)
    return {
        "short": duration < SHORT_TRIP_MAX_MIN,
        "long": duration > LONG_TRIP_MIN_MIN,
        "airport": airport.to_numpy(),
        "uncommon_pair": seen < UNCOMMON_PAIR_MAX_TRIPS,
    }


def segment_metrics(eval_df: pd.DataFrame, y_pred, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """RMSE and MAE per segment, e.g. {'short_rmse': ..., 'short_mae': ..., ...}."""
    y = eval_df["duration"].to_numpy()
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {}
    for name, mask in masks.items():
        if mask.any():
            out.update(regression_metrics(y[mask], y_pred[mask], prefix=f"{name}_"))
        else:
            out.update({f"{name}_rmse": float("nan"), f"{name}_mae": float("nan")})
    return out


def segment_sizes(masks: dict[str, np.ndarray]) -> pd.Series:
    """Number of trips in each segment."""
    return pd.Series({name: int(mask.sum()) for name, mask in masks.items()}, name="trips")


def score_predictions(split: str, train_df: pd.DataFrame, eval_df: pd.DataFrame, pred) -> dict[str, float]:
    """<split>_rmse/mae and <split>_<segment>_rmse/mae (segments use training-period pair counts)."""
    masks = segment_masks(eval_df, zone_pair_counts(train_df))
    metrics = regression_metrics(eval_df["duration"], pred, prefix=f"{split}_")
    metrics.update({f"{split}_{k}": v for k, v in segment_metrics(eval_df, pred, masks).items()})
    return metrics


def describe_period(df: pd.DataFrame) -> str:
    """'2025-01-01 .. 2025-01-20 (2,061,234 trips)' for a frame of trips."""
    pickup = df["tpep_pickup_datetime"]
    return f"{pickup.min():%Y-%m-%d} .. {pickup.max():%Y-%m-%d} ({len(df):,} trips)"


def fit_rf(params: dict, X: pd.DataFrame, y: pd.Series):
    """Fit a RandomForestRegressor with the project's fixed seed (all CPU cores)."""
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE).fit(X, y)


def model_size_mb(model) -> float:
    """Size of the model serialized with skops (the format logged to MLflow), in MB (MiB)."""
    import skops.io as sio

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.skops"
        sio.dump(model, path)
        return path.stat().st_size / 2**20
