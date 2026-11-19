"""Hyperparameter tuning with Optuna, logged to MLflow as nested runs."""

import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from src.experiment import TRACKING_URI


def tune_random_forest(X_train, y_train, X_val, y_val, n_trials=20, seed=42,
                       experiment="nyc-taxi-duration"):
    """Search Random Forest settings on validation RMSE.

    Each trial is logged as a child run under one parent run "optuna-tuning".
    Returns the Optuna study (use study.best_params and study.best_value).
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
            model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
            rmse = root_mean_squared_error(y_val, model.predict(X_val))
            mlflow.log_params(params)
            mlflow.log_metric("val_rmse", rmse)
        return rmse

    with mlflow.start_run(run_name="optuna-tuning"):
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_rmse", study.best_value)

    print(f"Best RMSE: {study.best_value:.3f}")
    print("Best params:", study.best_params)
    return study