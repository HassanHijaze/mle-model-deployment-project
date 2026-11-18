"""Shared test setup (fixtures). pytest loads this file automatically.

Fixtures are created once per test session and skip their tests if the
data or the champion model is not available.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "nyc-taxi-duration-rf"
MODEL_ALIAS = "champion"


@pytest.fixture(scope="session")
def data():
    from src.train import load_data
    try:
        return load_data()
    except Exception as e:
        pytest.skip(f"Could not load the taxi data: {e}")


@pytest.fixture(scope="session")
def zone_xy():
    from src.features import build_zone_xy
    return build_zone_xy(ROOT / "data")


@pytest.fixture(scope="session")
def champion_version():
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    try:
        return str(MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)
    except Exception:
        pytest.skip("Register a model and set the 'champion' alias in the MLflow UI")


@pytest.fixture(scope="session")
def model(champion_version):
    import mlflow.sklearn
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champion_version}")


@pytest.fixture(scope="session")
def client(champion_version):
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
