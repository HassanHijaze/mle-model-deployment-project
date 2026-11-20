"""Train a Random Forest and log the run to the MLflow server.

The server address comes from the MLFLOW_TRACKING_URI environment variable
(default: http://localhost:5002, the port mapped in docker-compose.yml).
"""

import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, root_mean_squared_log_error

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
EXPERIMENT = "nyc-taxi-duration"
ZONE_FILE = "zone_xy.csv"  # zone table logged with every model, loaded by the API


def fit_and_score(params, X_train, y_train, X_eval, y_eval, eval_name="val"):
    """Train a Random Forest and score it on (X_eval, y_eval). Nothing is logged.

    Returns the fitted model and a dict of metrics (RMSE, MAE, RMSLE).
    """
    model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    metrics = {
        f"{eval_name}_rmse": root_mean_squared_error(y_eval, y_pred),
        f"{eval_name}_mae": mean_absolute_error(y_eval, y_pred),
        f"{eval_name}_rmsle": root_mean_squared_log_error(y_eval, y_pred),
    }
    print(eval_name, {k: round(v, 4) for k, v in metrics.items()})
    return model, metrics


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


def log_model_run(
    model,
    params,
    metrics,
    features,
    run_name,
    experiment=EXPERIMENT,
    register_name=None,
    zone_xy=None,
    X=None,
    training_period=None,
):
    """Log an already trained model to MLflow in a new run, with its params and metrics.

    See log_model_artifacts() for the zone table, the bundle metadata (X,
    training_period) and registration (no alias is set).
    Returns the registered version (str) or None.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_metrics(metrics)
        version = log_model_artifacts(
            model, zone_xy=zone_xy, register_name=register_name, X=X, training_period=training_period
        )

    print(run_name, {k: round(v, 4) for k, v in metrics.items()})
    return version


def train_and_log(
    params,
    X_train,
    y_train,
    X_eval,
    y_eval,
    run_name,
    eval_name="val",
    experiment=EXPERIMENT,
    register_name=None,
    zone_xy=None,
):
    """Fit the model, score it on (X_eval, y_eval), and log everything to MLflow.

    Returns the fitted model and a dict of metrics.
    """
    model, metrics = fit_and_score(params, X_train, y_train, X_eval, y_eval, eval_name)
    log_model_run(
        model, params, metrics, X_train.columns, run_name, experiment, register_name, zone_xy, X=X_train
    )
    return model, metrics
