"""Model bundle: a registered model together with everything needed to serve it.

A bundle is loaded and checked as ONE unit, by the API (/reload, startup) and by
`nyc-taxi verify <version>`:

    model           the scikit-learn model (MLflow artifact "model")
    zone table      zone_xy.csv, logged in the same run (zone centers used by the features)
    feature schema  feature_schema.json: feature names, order and dtypes at training time
    metadata        run params: training period and library versions at training time

log_bundle_metadata() writes the schema, training period and versions when a model is
logged (src.experiment.log_model_artifacts). validate_bundle() checks a loaded bundle
before it is served: schema == src.features.FEATURES, zone table present, and test
predictions on REFERENCE_TRIPS are finite and within the trained 1-60 minute range.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import FEATURES, build_features

ZONE_FILE = "zone_xy.csv"
FEATURE_SCHEMA_FILE = "feature_schema.json"
ZONE_COLUMNS = ("lat", "lon", "x", "y")
MIN_MINUTES, MAX_MINUTES = 1.0, 60.0  # the training target range (src.train.clean_trips)

# Known trips predicted on every check: (pickup zone, drop-off zone, pickup time).
# 161 Midtown Center, 236 Upper East Side North, 132 JFK Airport, 74 East Harlem North,
# 162 Midtown East, 264 Unknown (no zone shape: features -1).
REFERENCE_TRIPS = [
    (161, 236, "2025-01-27 08:30"),
    (132, 161, "2025-01-28 17:00"),
    (74, 162, "2025-01-01 12:00"),
    (161, 264, "2025-01-30 10:00"),
]

# Logged as run params lib_<name> (training environment)
LIBRARIES = {"scikit_learn": "scikit-learn", "pandas": "pandas", "numpy": "numpy", "mlflow": "mlflow"}


class BundleLoadError(RuntimeError):
    """The bundle could not be fetched (registry unreachable, unknown version, broken model)."""


@dataclass(frozen=True)
class Bundle:
    model: Any
    zone_xy: pd.DataFrame | None
    schema: dict | None
    version: str
    run_id: str | None = None
    params: dict = field(default_factory=dict)       # run params (training period, versions)
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------------
# Writing (training side)
# ---------------------------------------------------------------------------------


def feature_schema(X: pd.DataFrame) -> dict:
    """{"features": [{"name": ..., "dtype": ...}, ...]} in column order."""
    return {"features": [{"name": str(c), "dtype": str(X[c].dtype)} for c in X.columns]}


def library_versions() -> dict[str, str]:
    """Installed versions of the libraries that read and run the model."""
    out = {}
    for key, dist in LIBRARIES.items():
        try:
            out[key] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            out[key] = "not installed"
    return out


def reference_frame() -> pd.DataFrame:
    """REFERENCE_TRIPS as the raw columns build_features expects."""
    return pd.DataFrame([{"PULocationID": pu, "DOLocationID": do,
                          "tpep_pickup_datetime": pd.Timestamp(when)}
                         for pu, do, when in REFERENCE_TRIPS])


def log_bundle_metadata(X: pd.DataFrame, training_period: tuple[str, str] | None = None) -> None:
    """Log feature_schema.json, the training period and library versions into the ACTIVE run.

    X: the training features (only column names and dtypes are used).
    training_period: (first pickup date, last pickup date) of the trips the model was fitted on.
    """
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / FEATURE_SCHEMA_FILE
        path.write_text(json.dumps(feature_schema(X), indent=2), encoding="utf-8")
        mlflow.log_artifact(str(path))
    params = {f"lib_{k}": v for k, v in library_versions().items()}
    if training_period is not None:
        params["bundle_training_start"], params["bundle_training_end"] = training_period
    mlflow.log_params(params)


def period_of(df: pd.DataFrame) -> tuple[str, str]:
    """(first, last) pickup date of a frame of trips, as YYYY-MM-DD."""
    pickup = df["tpep_pickup_datetime"]
    return f"{pickup.min():%Y-%m-%d}", f"{pickup.max():%Y-%m-%d}"


# ---------------------------------------------------------------------------------
# Reading (serving side). The three helpers below are the only MLflow calls, so tests
# can replace them.
# ---------------------------------------------------------------------------------


def _client():
    from mlflow import MlflowClient

    return MlflowClient()


def _load_model(uri: str):
    import mlflow.sklearn

    return mlflow.sklearn.load_model(uri)


def _download(run_id: str, artifact_path: str) -> str:
    import mlflow.artifacts

    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)


def fetch_bundle(model_name: str, *, alias: str | None = None, version: str | int | None = None) -> Bundle:
    """Load one registered version (by alias or number) with its zone table and schema.

    Raises BundleLoadError if the registry is unreachable, the version/alias does not
    exist or the model cannot be loaded. A missing zone table or schema does not raise:
    the bundle is returned without it and validate_bundle() reports the problem.
    """
    if (alias is None) == (version is None):
        raise ValueError("give exactly one of alias or version")
    which = f"alias '{alias}'" if alias else f"version {version}"
    try:
        client = _client()
        mv = (client.get_model_version_by_alias(model_name, alias) if alias
              else client.get_model_version(model_name, str(version)))
    except Exception as exc:
        raise BundleLoadError(f"MLflow registry unreachable or {model_name} {which} not found: "
                              f"{type(exc).__name__}: {exc}") from exc

    version = str(mv.version)
    try:
        model = _load_model(f"models:/{model_name}/{version}")
    except Exception as exc:
        raise BundleLoadError(f"{model_name} version {version} could not be loaded: "
                              f"{type(exc).__name__}: {exc}") from exc

    warnings = []
    zone_xy = None
    try:
        zone_xy = pd.read_csv(_download(mv.run_id, ZONE_FILE), index_col="LocationID")
    except Exception as exc:
        warnings.append(f"zone table {ZONE_FILE} unavailable: {type(exc).__name__}: {exc}")

    schema = None
    try:
        schema = json.loads(Path(_download(mv.run_id, FEATURE_SCHEMA_FILE)).read_text(encoding="utf-8"))
    except Exception:
        warnings.append(f"{FEATURE_SCHEMA_FILE} not logged with this model (older model): "
                        "checked against the model's own feature names instead")

    params = {}
    try:
        params = dict(client.get_run(mv.run_id).data.params)
    except Exception as exc:
        warnings.append(f"run params unavailable: {type(exc).__name__}: {exc}")

    return Bundle(model=model, zone_xy=zone_xy, schema=schema, version=version, run_id=mv.run_id,
                  params=params, warnings=tuple(warnings))


# ---------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------


def predict_reference_trips(model, zone_xy: pd.DataFrame) -> np.ndarray:
    """Predictions (minutes) for REFERENCE_TRIPS, built with the API's feature code."""
    return np.asarray(model.predict(build_features(reference_frame(), zone_xy)), dtype=float)


def validate_bundle(bundle: Bundle) -> list[str]:
    """Problems that make the bundle unsafe to serve ([] = OK)."""
    problems = []

    # Feature schema: names and order must be FEATURES; dtypes must be numeric
    if bundle.schema is not None:
        try:
            names = [f["name"] for f in bundle.schema["features"]]
            dtypes = {f["name"]: f["dtype"] for f in bundle.schema["features"]}
        except (KeyError, TypeError):
            problems.append(f"{FEATURE_SCHEMA_FILE} is malformed (expected a 'features' list of name/dtype)")
            names, dtypes = None, {}
        if names is not None and names != FEATURES:
            problems.append(f"feature schema does not match FEATURES: expected {FEATURES}, got {names}")
        for name, dtype in dtypes.items():
            try:
                numeric = np.dtype(dtype).kind in "iufb"
            except TypeError:
                numeric = False
            if not numeric:
                problems.append(f"feature '{name}' has non-numeric dtype {dtype!r} in the schema")

    model_names = getattr(bundle.model, "feature_names_in_", None)
    if model_names is not None and list(model_names) != FEATURES:
        problems.append(f"model was trained on features {list(model_names)}, the API builds {FEATURES}")
    if bundle.schema is None and model_names is None:
        problems.append(f"{FEATURE_SCHEMA_FILE} missing and the model has no feature names to check")

    # Zone table
    zones = bundle.zone_xy
    if zones is None or len(zones) == 0:
        problems.append(f"zone table {ZONE_FILE} missing or empty")
    elif missing := [c for c in ZONE_COLUMNS if c not in zones.columns]:
        problems.append(f"zone table {ZONE_FILE} lacks columns {missing}")

    # Test predictions (only when the inputs are usable)
    if not problems:
        try:
            pred = predict_reference_trips(bundle.model, zones)
        except Exception as exc:
            problems.append(f"test prediction failed: {type(exc).__name__}: {exc}")
        else:
            if pred.shape != (len(REFERENCE_TRIPS),):
                problems.append(f"test prediction returned shape {pred.shape}, "
                                f"expected ({len(REFERENCE_TRIPS)},)")
            elif not np.all(np.isfinite(pred)):
                problems.append(f"test predictions are not all finite: {pred.tolist()}")
            elif np.any(pred < MIN_MINUTES) or np.any(pred > MAX_MINUTES):
                problems.append(f"test predictions outside {MIN_MINUTES:g}-{MAX_MINUTES:g} min: "
                                f"{np.round(pred, 2).tolist()}")
    return problems


def version_differences(params: dict) -> list[str]:
    """Libraries whose installed version differs from the one logged at training time."""
    installed = library_versions()
    return [f"{dist}: trained with {params[f'lib_{key}']}, installed {installed[key]}"
            for key, dist in LIBRARIES.items()
            if f"lib_{key}" in params and params[f"lib_{key}"] != installed[key]]
