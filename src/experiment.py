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
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
ZONE_FILE = "zone_xy.csv"   # zone table logged with every model, loaded by the API


def fit_and_score(params, X_train, y_train, X_eval, y_eval, eval_name="val"):
    """Train a Random Forest and score it on (X_eval, y_eval). Nothing is logged.

    Returns the fitted model and a dict of metrics.
    """
    model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    metrics = {
        f"{eval_name}_rmse": root_mean_squared_error(y_eval, y_pred),
        f"{eval_name}_rmsle": root_mean_squared_log_error(y_eval, y_pred),
    }
    print(eval_name, {k: round(v, 4) for k, v in metrics.items()})
    return model, metrics


def log_model_run(model, params, metrics, features, run_name,
                  experiment="nyc-taxi-duration", register_name=None, zone_xy=None):
    """Log an already trained model to MLflow (as skops), with its params and metrics.

    If zone_xy is given, the zone table is logged in the same run as zone_xy.csv,
    so the API can load exactly the zone map the model was trained with.
    If register_name is given, the model is registered as a new version.
    No alias is set: choose the champion yourself in the MLflow UI.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_metrics(metrics)
        if zone_xy is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / ZONE_FILE
                zone_xy.to_csv(path)
                mlflow.log_artifact(str(path))
        model_info = mlflow.sklearn.log_model(
            model, name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            registered_model_name=register_name,
        )

    print(run_name, {k: round(v, 4) for k, v in metrics.items()})
    if register_name:
        print(f"Registered {register_name} version {model_info.registered_model_version} "
              f"(no alias set - set 'champion' in the MLflow UI if you want to serve it)")


def train_and_log(params, X_train, y_train, X_eval, y_eval, run_name,
                  eval_name="val", experiment="nyc-taxi-duration", register_name=None,
                  zone_xy=None):
    """Fit the model, score it on (X_eval, y_eval), and log everything to MLflow.

    Returns the fitted model and a dict of metrics.
    """
    model, metrics = fit_and_score(params, X_train, y_train, X_eval, y_eval, eval_name)
    log_model_run(model, params, metrics, X_train.columns, run_name,
                  experiment, register_name, zone_xy)
    return model, metrics