"""Hyperparameter tuning with Optuna, logged to MLflow as nested runs.

    uv run nyc-taxi tune      (or uv run python -m src.tuning)

Runs 20 TPE trials on the 500k-trip sample (the same sample as rf-baseline): fit on the
sample's training days (Jan 1-20), minimise RMSE on its validation days (Jan 21-25).
The test days are never used. Logged as one parent run "optuna-tuning" with one child
run per trial; the parent holds best_<param> params and the best_val_rmse metric, which
`nyc-taxi train` and `select` read (src.train.load_tuned_params).

Runtime: roughly 15-40 min (all CPU cores), < 3 GB RAM. The MLflow server must be running.
"""

import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from src.experiment import EXPERIMENT, TRACKING_URI

N_TRIALS = 20


def tune_random_forest(
    X_train, y_train, X_val, y_val, n_trials=N_TRIALS, seed=42, experiment=EXPERIMENT, extra_params=None
):
    """Search Random Forest settings on validation RMSE.

    Each trial is logged as a child run under one parent run "optuna-tuning".
    extra_params (optional) are logged on the parent, e.g. the sample size.
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
        if extra_params:
            mlflow.log_params(extra_params)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_rmse", study.best_value)

    print(f"Best RMSE: {study.best_value:.3f}")
    print("Best params:", study.best_params)
    return study


def main(n_trials: int = N_TRIALS) -> dict:
    """Tune on the 500k sample (train Jan 1-20, validate Jan 21-25) and log to MLflow.

    Returns {"best_params": ..., "best_val_rmse": ...}.
    """
    from src.features import build_features, build_zone_xy
    from src.train import BEST_PARAMS, ROOT, SAMPLE_SIZE, load_data, split_by_time

    sample = load_data(SAMPLE_SIZE)  # same 500k sample as rf-baseline
    X, y = build_features(sample, build_zone_xy(ROOT / "data")), sample["duration"]
    tr, va, _ = split_by_time(sample)
    print(f"Sample: train {tr.sum():,}  validation {va.sum():,} (test days not used)")

    study = tune_random_forest(
        X[tr],
        y[tr],
        X[va],
        y[va],
        n_trials=n_trials,
        extra_params={"sample_size": len(sample), "n_trials": n_trials},
    )
    if study.best_params != BEST_PARAMS:
        print(
            f"Note: the best params differ from the fallback BEST_PARAMS in src/train.py "
            f"{BEST_PARAMS}; train and select use the new ones from MLflow."
        )
    return {"best_params": study.best_params, "best_val_rmse": study.best_value}


if __name__ == "__main__":
    main()
