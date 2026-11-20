"""The `nyc-taxi tune` step: Optuna search, logged to MLflow as nested runs.

    uv run nyc-taxi tune      (or uv run python -m src.tuning)

20 TPE trials. Each trial fits a Random Forest on the training sample (Jan 1-20, see
src.train.training_sample) and scores it on all validation trips (Jan 21-25). The test
days are never used. Logged as one parent run "optuna-tuning" with one child run per
trial (params, val_rmse, val_mae, model_mb); the parent holds best_<param> params and
best_val_rmse. `nyc-taxi select` reads every trial.
"""

from __future__ import annotations

import mlflow
import optuna

from src import evaluation as ev
from src.experiment import EXPERIMENT, TRACKING_URI

N_TRIALS = 20
TUNING_RUN_NAME = "optuna-tuning"


def tune_random_forest(
    X_train, y_train, X_val, y_val, n_trials=N_TRIALS, seed=42, experiment=EXPERIMENT, extra_params=None
):
    """Search Random Forest settings on validation RMSE.

    Each trial is a child run of "optuna-tuning" logging its params, val_rmse, val_mae and
    model_mb (skops size). extra_params (optional) are logged on the parent.
    Returns the Optuna study (study.best_params, study.best_value).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [1.0, 0.7, 0.5]),
        }
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            model = ev.fit_rf(params, X_train, y_train)
            metrics = ev.regression_metrics(y_val, model.predict(X_val), prefix="val_")
            metrics["model_mb"] = ev.model_size_mb(model)
            del model
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        return metrics["val_rmse"]

    with mlflow.start_run(run_name=TUNING_RUN_NAME):
        if extra_params:
            mlflow.log_params(extra_params)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_rmse", study.best_value)

    print(f"Best validation RMSE: {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return study


def main(n_trials: int = N_TRIALS) -> dict:
    """Tune on the training sample (Jan 1-20), score on validation (Jan 21-25), log to MLflow."""
    from src.features import build_features, build_zone_xy
    from src.train import ROOT, load_splits, period_params

    splits = load_splits()
    zone_xy = build_zone_xy(ROOT / "data")
    X_train, y_train = build_features(splits.sample, zone_xy), splits.sample["duration"]
    X_val, y_val = build_features(splits.val, zone_xy), splits.val["duration"]
    print(f"Training sample: {len(X_train):,} trips (Jan 1-20); validation: {len(X_val):,} trips (Jan 21-25)")

    study = tune_random_forest(
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=n_trials,
        extra_params={"n_trials": n_trials, **period_params(splits)},
    )
    return {"best_params": study.best_params, "best_val_rmse": study.best_value}


if __name__ == "__main__":
    main()
