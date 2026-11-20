"""End-to-end test of the deployed stack: postgres + mlflow + api with docker compose.

    uv run pytest -m e2e tests/e2e

Starts a THROWAWAY stack under its own compose project name ("nyc-taxi-e2e") and its own
ports, so it never touches the development stack or its registry. No real training: it
registers tiny deterministic synthetic models in the same bundle format as training
(src.experiment.log_model_artifacts: model + zone_xy.csv + feature_schema.json), then:

    1. version 1 -> champion -> /reload -> /ready and /predict answer with version 1
    2. version 2 -> champion -> /reload -> version 2 answers
    3. broken version 3 -> champion -> /reload is rejected -> version 2 still answers
    4. /metrics shows the served version, predictions and the rejected reload

The containers and volumes are always removed at the end (docker compose down -v).
Settings come from the environment (CI sets test values): E2E_ADMIN_TOKEN,
E2E_POSTGRES_USER, E2E_POSTGRES_PASSWORD, E2E_MLFLOW_PORT, E2E_API_PORT.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from src.experiment import MODEL_NAME

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "nyc-taxi-e2e"
MLFLOW_PORT = os.getenv("E2E_MLFLOW_PORT", "15002")
API_PORT = os.getenv("E2E_API_PORT", "18000")
ADMIN_TOKEN = os.getenv("E2E_ADMIN_TOKEN", "e2e-admin-token")
MLFLOW_URI = f"http://127.0.0.1:{MLFLOW_PORT}"
API_URL = f"http://127.0.0.1:{API_PORT}"
ARTIFACTS_DIR = ROOT / ".e2e-mlartifacts"
TRIP = {"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}
AUTH = {"X-Admin-Token": ADMIN_TOKEN}


def compose(*args: str, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", "-p", PROJECT, *args], cwd=ROOT, env=env, check=check)


@pytest.fixture(scope="module")
def stack():
    """Bring the throwaway stack up; always tear it down (containers and volumes)."""
    if shutil.which("docker") is None:
        pytest.skip("docker is not available")
    import mlflow

    assert MLFLOW_PORT != "5002" and API_PORT != "8000", "e2e must not use the development ports"
    env = {
        **os.environ,
        "POSTGRES_USER": os.getenv("E2E_POSTGRES_USER", "e2e"),
        "POSTGRES_PASSWORD": os.getenv("E2E_POSTGRES_PASSWORD", "e2e-password"),
        "POSTGRES_DB": "mlflow_e2e",
        "MLFLOW_VERSION": mlflow.__version__,  # server image = client version from uv.lock
        "ADMIN_TOKEN": ADMIN_TOKEN,
        "MLFLOW_PORT": MLFLOW_PORT,
        "API_PORT": API_PORT,
        "MLFLOW_ARTIFACTS_DIR": str(ARTIFACTS_DIR),
    }
    compose("down", "-v", "--remove-orphans", env=env, check=False)
    shutil.rmtree(ARTIFACTS_DIR, ignore_errors=True)
    ARTIFACTS_DIR.mkdir()
    try:
        compose(
            "up", "-d", "--build", "--wait", "--wait-timeout", "600", "postgres", "mlflow", "api", env=env
        )
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("e2e")
        yield
    except BaseException:
        compose("logs", "--no-color", "--tail", "200", env=env, check=False)
        raise
    finally:
        compose("down", "-v", "--remove-orphans", env=env, check=False)
        shutil.rmtree(ARTIFACTS_DIR, ignore_errors=True)


def synthetic_trips(n: int = 400) -> pd.DataFrame:
    """Deterministic trips between the zones of the fake zone table."""
    from tests.fakes import fake_zones

    rng = np.random.default_rng(0)
    zones = fake_zones().index.to_numpy()
    pickup = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 31 * 24 * 60, n), unit="min")
    return pd.DataFrame(
        {
            "PULocationID": rng.choice(zones, n),
            "DOLocationID": rng.choice(zones, n),
            "tpep_pickup_datetime": pickup,
        }
    )


def register_synthetic_model(offset_minutes: float, drop_feature: bool = False) -> str:
    """Fit a tiny forest on synthetic trips and register it like training does. Returns the version."""
    import mlflow
    from sklearn.ensemble import RandomForestRegressor

    from src.experiment import log_model_artifacts
    from src.features import FEATURES, build_features
    from tests.fakes import fake_zones

    zones = fake_zones()
    X = build_features(synthetic_trips(), zones)
    y = (5 + offset_minutes + 2 * X["straight_dist"].clip(lower=0) + 0.1 * X["hour"]).clip(2, 55)
    features = FEATURES[:-1] if drop_feature else FEATURES  # broken: one feature missing
    model = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0).fit(X[features], y)
    with mlflow.start_run(run_name=f"e2e-offset-{offset_minutes}{'-broken' if drop_feature else ''}"):
        version = log_model_artifacts(
            model,
            zone_xy=zones,
            register_name=MODEL_NAME,
            X=X[features],
            training_period=("2025-01-01", "2025-01-31"),
        )
    assert version is not None
    return version


def set_champion(version: str) -> None:
    from mlflow import MlflowClient

    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)  # throwaway registry only


def reload_api() -> requests.Response:
    return requests.post(f"{API_URL}/reload", headers=AUTH, timeout=300)


def predict() -> dict:
    r = requests.post(f"{API_URL}/predict", json=TRIP, timeout=30)
    assert r.status_code == 200, r.text
    assert r.headers["X-Request-ID"]
    return r.json()


def wait_until(check, timeout_s: float = 60.0):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if check():
                return
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(1)


def test_promote_upgrade_and_reject_broken_version(stack):
    wait_until(lambda: requests.get(f"{API_URL}/live", timeout=5).status_code == 200)
    not_ready = requests.get(f"{API_URL}/ready", timeout=5)
    assert not_ready.status_code == 503 and not_ready.json()["code"] == "not_ready"

    # 1. First version becomes champion
    v1 = register_synthetic_model(offset_minutes=0)
    set_champion(v1)
    assert requests.post(f"{API_URL}/reload", timeout=30).status_code == 401  # token required
    r = reload_api()
    assert r.status_code == 200, r.text
    assert r.json()["current_version"] == v1
    assert requests.get(f"{API_URL}/ready", timeout=5).json() == {"status": "ready", "model_version": v1}
    first = predict()
    assert first["model_version"] == v1 and 1 <= first["duration_minutes"] <= 60

    # 2. A second version replaces it
    v2 = register_synthetic_model(offset_minutes=10)
    set_champion(v2)
    r = reload_api()
    assert r.status_code == 200, r.text
    assert r.json()["previous_version"] == v1 and r.json()["current_version"] == v2
    second = predict()
    assert second["model_version"] == v2
    assert second["duration_minutes"] != first["duration_minutes"]

    # 3. A broken version is rejected and the previous one keeps answering
    v3 = register_synthetic_model(offset_minutes=20, drop_feature=True)
    set_champion(v3)
    r = reload_api()
    body = r.json()
    assert r.status_code == 422, r.text
    assert body["code"] == "bundle_invalid"
    assert body["rejected_version"] == v3 and body["serving_version"] == v2
    assert predict()["model_version"] == v2
    assert requests.get(f"{API_URL}/ready", timeout=5).json()["model_version"] == v2

    # 4. Prometheus metrics: served version, predictions and the rejected reload are visible
    m = requests.get(f"{API_URL}/metrics", timeout=10)
    assert m.status_code == 200
    assert f'taxi_api_model_info{{model_version="{v2}"}} 1.0' in m.text
    assert "taxi_api_predictions_total" in m.text
    assert 'taxi_api_reload_failures_total{reason="invalid_bundle"} 1.0' in m.text
