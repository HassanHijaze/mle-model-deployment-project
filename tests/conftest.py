"""Shared test setup (fixtures). pytest loads this file automatically.

Fixtures are created once per test session and skip their tests if the
data, the zone map or the champion model is not available.

The model and API tests need the MLflow server running (docker compose up -d);
its address comes from MLFLOW_TRACKING_URI (default: http://localhost:5002).
"""

import os
from pathlib import Path

import pytest

from src.experiment import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]
MODEL_ALIAS = "champion"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")


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

    try:
        return build_zone_xy(ROOT / "data")
    except Exception as e:
        pytest.skip(f"Could not build the zone map: {e}")


@pytest.fixture(scope="session")
def champion_version():
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        return str(MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)
    except Exception:
        pytest.skip(
            "No champion model found - register a model and set the "
            "'champion' alias in the MLflow UI (and make sure the MLflow server is running)"
        )


@pytest.fixture(scope="session")
def model(champion_version):
    import mlflow.sklearn

    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champion_version}")


@pytest.fixture(scope="session")
def client(champion_version):
    from fastapi.testclient import TestClient

    from app.main import app

    # "with" runs the app's startup code, which loads the zone map and the champion model
    with TestClient(app) as c:
        yield c
