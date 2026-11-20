"""MLflow settings and logging a model as a complete bundle.

The server address comes from the MLFLOW_TRACKING_URI environment variable
(default: http://localhost:5002, the port mapped in docker-compose.yml).
"""

import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
EXPERIMENT = "nyc-taxi-duration"
MODEL_NAME = "nyc-taxi-duration-rf"  # registered model; the API serves its "champion" version
MISSING_RUN_HINT = "Run `uv run nyc-taxi all` first (MLflow server must be running)."
ZONE_FILE = "zone_xy.csv"  # zone table logged with every model, loaded by the API


def log_model_artifacts(model, zone_xy=None, register_name=None, X=None, training_period=None):
    """Log a trained model (skops) into the ACTIVE MLflow run, as a complete bundle.

    If zone_xy is given, it is logged in the same run as zone_xy.csv, so the API can
    load exactly the zone map the model was trained with. If X (the training features)
    is given, feature_schema.json (names, order, dtypes), the training period
    (first, last pickup date) and the installed library versions are logged too
    (src.bundle.log_bundle_metadata). If register_name is given, the model is
    registered as a new version WITHOUT an alias (the "champion" alias is only moved by
    hand). Returns the registered version (str) or None.
    """
    from src.bundle import log_bundle_metadata

    if X is not None:
        log_bundle_metadata(X, training_period)
    if zone_xy is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ZONE_FILE
            zone_xy.to_csv(path)
            mlflow.log_artifact(str(path))
    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
        registered_model_name=register_name,
    )
    if not register_name:
        return None
    version = str(model_info.registered_model_version)
    print(
        f"Registered {register_name} version {version} "
        f"(no alias set - set 'champion' in the MLflow UI if you want to serve it)"
    )
    return version
