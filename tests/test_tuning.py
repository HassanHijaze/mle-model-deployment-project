"""Each Optuna trial logs params, val_rmse, val_mae and model_mb (MLflow mocked, tiny data)."""

from contextlib import contextmanager

import numpy as np
import pandas as pd

from src import evaluation as ev
from src import tuning
from src.features import FEATURES


def tiny_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(FEATURES))), columns=FEATURES)
    y = pd.Series(5 + 20 * X["straight_dist"] + rng.random(n))
    return X, y


def test_every_trial_logs_params_metrics_and_model_size(monkeypatch):
    runs, active = [], []  # every run started; the stack of open runs

    @contextmanager
    def start_run(run_name=None, nested=False):
        run = {"name": run_name, "nested": nested, "params": {}, "metrics": {}}
        runs.append(run)
        active.append(run)
        try:
            yield
        finally:
            active.pop()

    def current():
        return active[-1]

    monkeypatch.setattr(tuning.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(tuning.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(tuning.mlflow, "start_run", start_run)
    monkeypatch.setattr(tuning.mlflow, "log_params", lambda p: current()["params"].update(p))
    monkeypatch.setattr(tuning.mlflow, "log_metrics", lambda m: current()["metrics"].update(m))
    monkeypatch.setattr(tuning.mlflow, "log_metric", lambda k, v: current()["metrics"].update({k: v}))
    X, y = tiny_data()
    X_val, y_val = tiny_data(seed=1)

    study = tuning.tune_random_forest(X, y, X_val, y_val, n_trials=2)

    trials = [r for r in runs if r["nested"]]
    assert [t["name"] for t in trials] == ["trial-0", "trial-1"]
    for t in trials:
        assert set(t["params"]) == {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
        assert {"val_rmse", "val_mae", "model_mb"} <= set(t["metrics"])
        assert t["metrics"]["model_mb"] > 0
    assert runs[0]["name"] == "optuna-tuning" and not runs[0]["nested"]
    assert study.best_value == min(t["metrics"]["val_rmse"] for t in trials)


def test_model_size_mb_is_positive_and_grows_with_trees():
    X, y = tiny_data()
    small = ev.fit_rf({"n_estimators": 2, "max_depth": 3}, X, y)
    large = ev.fit_rf({"n_estimators": 20, "max_depth": 3}, X, y)
    assert 0 < ev.model_size_mb(small) < ev.model_size_mb(large)


def test_same_params_same_data_same_model():
    X, y = tiny_data()
    params = {"n_estimators": 5, "max_depth": 4, "max_features": 0.5}
    a, b = ev.fit_rf(params, X, y), ev.fit_rf(params, X, y)
    for tree_a, tree_b in zip(a.estimators_, b.estimators_, strict=True):  # identical trees
        np.testing.assert_array_equal(tree_a.tree_.threshold, tree_b.tree_.threshold)
        np.testing.assert_array_equal(tree_a.tree_.value, tree_b.tree_.value)
    # predictions: equal up to float summation order (n_jobs=-1 adds the trees in thread order)
    np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=0, atol=1e-9)
