"""Safe hot reload: a reload is validated before the swap, and the old model keeps serving
when it fails. Fake bundles only - no MLflow server, no real model (runs in CI)."""

import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from src import bundle as bundle_lib
from tests.fakes import FakeMlflowClient, FakeModel, make_bundle

TRIP = {"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}
TOKEN = "test-admin-token"
AUTH = {"X-Admin-Token": TOKEN}
REAL_LOAD_CHAMPION = main.load_champion


@pytest.fixture
def api(monkeypatch):
    """Client whose champion is version 1; set loader["load"] to change what /reload gets."""
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    loader = {"load": lambda: make_bundle("1")}
    monkeypatch.setattr(main, "load_champion", lambda: loader["load"]())
    with TestClient(main.app) as client:
        yield client, loader


def serving_version(client):
    r = client.post("/predict", json=TRIP)
    assert r.status_code == 200
    assert r.headers["X-Model-Version"] == r.json()["model_version"]
    return r.json()["model_version"]


def test_startup_serves_the_champion(api):
    client, _ = api
    assert client.get("/ready").json() == {"status": "ready", "model_version": "1"}
    assert serving_version(client) == "1"


def test_successful_reload_swaps_the_model(api):
    client, loader = api
    loader["load"] = lambda: make_bundle("2", model=FakeModel(minutes=20.0))
    r = client.post("/reload", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["previous_version"] == "1" and r.json()["current_version"] == "2"
    assert serving_version(client) == "2"
    assert client.post("/predict", json=TRIP).json()["duration_minutes"] == 20.0


@pytest.mark.parametrize(
    "headers, env, status, code",
    [
        ({}, TOKEN, 401, "unauthorized"),
        ({"X-Admin-Token": "wrong"}, TOKEN, 401, "unauthorized"),
        (AUTH, None, 503, "reload_disabled"),
    ],
)
def test_reload_needs_the_admin_token(api, monkeypatch, headers, env, status, code):
    client, loader = api
    if env is None:
        monkeypatch.delenv("ADMIN_TOKEN")
    loader["load"] = lambda: make_bundle("2")
    r = client.post("/reload", headers=headers)
    assert r.status_code == status and r.json()["code"] == code
    assert serving_version(client) == "1"


def bad_schema_bundle():
    schema = make_bundle().schema
    schema["features"] = list(reversed(schema["features"]))
    return make_bundle("2", schema=schema)


@pytest.mark.parametrize(
    "bad_bundle, expected_problem",
    [
        (bad_schema_bundle, "feature schema does not match FEATURES"),
        (lambda: make_bundle("2", model=FakeModel(features=["a", "b"])), "model was trained on features"),
        (lambda: make_bundle("2", zone_xy=None), "zone table zone_xy.csv missing"),
        (
            lambda: make_bundle("2", model=FakeModel(fail=ValueError("corrupted tree"))),
            "test prediction failed",
        ),
        (lambda: make_bundle("2", model=FakeModel(minutes=np.nan)), "not all finite"),
        (lambda: make_bundle("2", model=FakeModel(minutes=500.0)), "outside 1-60 min"),
    ],
)
def test_invalid_bundle_is_rejected_and_old_model_keeps_serving(api, bad_bundle, expected_problem):
    client, loader = api
    loader["load"] = bad_bundle
    r = client.post("/reload", headers=AUTH)
    body = r.json()
    assert r.status_code == 422
    assert body["code"] == "bundle_invalid"
    assert body["rejected_version"] == "2" and body["serving_version"] == "1"
    assert any(expected_problem in p for p in body["problems"])
    assert serving_version(client) == "1"
    assert client.get("/ready").status_code == 200
    health = client.get("/health").json()
    assert health["last_reload"]["ok"] is False and expected_problem in health["last_reload"]["error"]


def test_mlflow_unreachable_keeps_old_model(api, monkeypatch):
    client, loader = api

    def unreachable():
        raise ConnectionError("Connection refused: localhost:5002")

    monkeypatch.setattr(bundle_lib, "_client", unreachable)
    loader["load"] = REAL_LOAD_CHAMPION  # the real loader, failing on the registry call
    r = client.post("/reload", headers=AUTH)
    assert r.status_code == 502
    assert r.json()["code"] == "reload_failed"
    assert "unreachable" in r.json()["message"] and r.json()["serving_version"] == "1"
    assert serving_version(client) == "1"


def test_corrupted_model_file_keeps_old_model(api, monkeypatch):
    client, loader = api
    monkeypatch.setattr(bundle_lib, "_client", lambda: FakeMlflowClient({"2": {}}, aliases={"champion": "2"}))

    def broken(uri):
        raise EOFError("unexpected end of skops file")

    monkeypatch.setattr(bundle_lib, "_load_model", broken)
    loader["load"] = REAL_LOAD_CHAMPION
    r = client.post("/reload", headers=AUTH)
    assert r.status_code == 502 and "could not be loaded" in r.json()["message"]
    assert serving_version(client) == "1"


def test_requests_during_a_slow_reload_use_the_old_model(api):
    client, loader = api
    started, release = threading.Event(), threading.Event()

    def slow_load():
        started.set()
        assert release.wait(timeout=10)
        return make_bundle("2")

    loader["load"] = slow_load
    result = {}
    thread = threading.Thread(target=lambda: result.update(r=client.post("/reload", headers=AUTH)))
    thread.start()
    try:
        assert started.wait(timeout=10)
        for _ in range(5):
            assert serving_version(client) == "1"
        second = client.post("/reload", headers=AUTH)
        assert second.status_code == 409 and second.json()["code"] == "reload_in_progress"
    finally:
        release.set()
        thread.join(timeout=10)
    assert result["r"].status_code == 200 and result["r"].json()["current_version"] == "2"
    assert serving_version(client) == "2"


def test_startup_without_model_then_reload(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)

    def no_champion():
        raise bundle_lib.BundleLoadError("no champion alias")

    loader = {"load": no_champion}
    monkeypatch.setattr(main, "load_champion", lambda: loader["load"]())
    with TestClient(main.app) as client:
        assert client.get("/live").json() == {"status": "alive"}
        not_ready = client.get("/ready")
        assert not_ready.status_code == 503 and not_ready.json()["code"] == "not_ready"
        assert "no champion alias" in not_ready.json()["message"]
        assert client.post("/predict", json=TRIP).json()["code"] == "model_not_loaded"

        loader["load"] = lambda: make_bundle("3")
        assert client.post("/reload", headers=AUTH).json()["current_version"] == "3"
        assert client.get("/ready").status_code == 200
        assert serving_version(client) == "3"


def test_invalid_bundle_at_startup_is_not_served(monkeypatch):
    monkeypatch.setattr(main, "load_champion", lambda: make_bundle("4", zone_xy=None))
    with TestClient(main.app) as client:
        r = client.get("/ready")
        assert r.status_code == 503 and "failed validation" in r.json()["message"]
        assert client.get("/live").status_code == 200
