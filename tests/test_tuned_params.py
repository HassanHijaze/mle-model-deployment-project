"""Unit tests for reading the tuned params from MLflow (MLflow is mocked, no training)."""

from types import SimpleNamespace

from src import train


def fake_run(params):
    return SimpleNamespace(info=SimpleNamespace(run_id="abc123"), data=SimpleNamespace(params=params))


def test_tuned_params_come_from_the_latest_tuning_run(monkeypatch, capsys):
    run = fake_run(
        {
            "best_n_estimators": "150",
            "best_max_depth": "30",
            "best_min_samples_leaf": "4",
            "best_max_features": "1.0",
            "sample_size": "500000",
        }
    )
    monkeypatch.setattr(train.ss, "latest_run", lambda name: run)
    params, source = train.load_tuned_params()
    assert params == {"n_estimators": 150, "max_depth": 30, "min_samples_leaf": 4, "max_features": 1.0}
    assert isinstance(params["max_features"], float)  # 1.0 = all features, 1 would mean one feature
    assert "abc123" in source
    assert "Params used" in capsys.readouterr().out


def test_fallback_to_best_params_without_a_tuning_run(monkeypatch, capsys):
    def no_run(name):
        raise LookupError(name)

    monkeypatch.setattr(train.ss, "latest_run", no_run)
    params, source = train.load_tuned_params()
    assert params == train.BEST_PARAMS
    assert "fallback" in source
    assert "nyc-taxi tune" in capsys.readouterr().out
