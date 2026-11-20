"""GET /metrics: Prometheus metrics of the API. Fake bundle only, no MLflow (runs in CI).

Counters are process-wide, so the tests compare values before and after a request.
"""

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import app.main as main
from app.observability import JsonFormatter
from src import bundle as bundle_lib
from tests.fakes import FakeModel, make_bundle

TRIP = {"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}
TOKEN = "metrics-test-token"
METRIC_NAMES = [
    "taxi_api_requests_total",
    "taxi_api_request_duration_seconds",
    "taxi_api_predictions_total",
    "taxi_api_prediction_errors_total",
    "taxi_api_model_info",
    "taxi_api_model_loaded",
    "taxi_api_model_load_duration_seconds",
    "taxi_api_reload_attempts_total",
    "taxi_api_reload_failures_total",
]


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    loader = {"load": lambda: make_bundle("5")}
    monkeypatch.setattr(main, "load_champion", lambda: loader["load"]())
    with TestClient(main.app) as client:
        yield client, loader


def test_metrics_endpoint_lists_every_custom_metric(api):
    client, _ = api
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    for name in METRIC_NAMES:
        assert f"# TYPE {name} " in r.text, name


def test_active_model_version_and_load_duration(api):
    assert value("taxi_api_model_info", model_version="5") == 1.0
    assert value("taxi_api_model_loaded") == 1.0
    assert value("taxi_api_model_load_duration_seconds_count", source="startup", outcome="ok") >= 1


def test_prediction_and_request_metrics(api):
    client, _ = api
    before_pred = value("taxi_api_predictions_total", model_version="5")
    before_req = value("taxi_api_requests_total", method="POST", route="/predict", status="200")
    before_lat = value("taxi_api_request_duration_seconds_count", method="POST", route="/predict")
    assert client.post("/predict", json=TRIP).status_code == 200
    assert value("taxi_api_predictions_total", model_version="5") == before_pred + 1
    assert value("taxi_api_requests_total", method="POST", route="/predict", status="200") == before_req + 1
    assert value("taxi_api_request_duration_seconds_count", method="POST", route="/predict") == before_lat + 1


def test_unknown_paths_share_one_route_label(api):
    client, _ = api
    before = value("taxi_api_requests_total", method="GET", route="unmatched", status="404")
    client.get("/no-such-page-1")
    client.get("/no-such-page-2")
    assert value("taxi_api_requests_total", method="GET", route="unmatched", status="404") == before + 2


def test_prediction_errors_are_counted(api):
    client, _ = api
    before = value("taxi_api_prediction_errors_total", reason="error")
    main.current_bundle().model.fail = RuntimeError("boom")
    assert client.post("/predict", json=TRIP).status_code == 500
    assert value("taxi_api_prediction_errors_total", reason="error") == before + 1


def test_reload_attempts_failures_and_version_switch(api):
    client, loader = api
    attempts = value("taxi_api_reload_attempts_total")
    invalid = value("taxi_api_reload_failures_total", reason="invalid_bundle")
    loader["load"] = lambda: make_bundle("6", model=FakeModel(minutes=500.0))  # fails validation
    assert client.post("/reload", headers={"X-Admin-Token": TOKEN}).status_code == 422
    assert value("taxi_api_reload_failures_total", reason="invalid_bundle") == invalid + 1
    assert value("taxi_api_model_info", model_version="5") == 1.0  # still serving 5

    loader["load"] = lambda: make_bundle("6")
    assert client.post("/reload", headers={"X-Admin-Token": TOKEN}).status_code == 200
    assert value("taxi_api_reload_attempts_total") == attempts + 2
    assert value("taxi_api_model_info", model_version="6") == 1.0
    assert REGISTRY.get_sample_value("taxi_api_model_info", {"model_version": "5"}) is None  # old series gone


def test_no_model_sets_loaded_to_zero(monkeypatch):
    def no_champion():
        raise bundle_lib.BundleLoadError("no champion alias")

    monkeypatch.setattr(main, "load_champion", no_champion)
    before = value("taxi_api_prediction_errors_total", reason="model_not_loaded")
    with TestClient(main.app) as client:
        assert value("taxi_api_model_loaded") == 0.0
        assert client.post("/predict", json=TRIP).status_code == 503
    assert value("taxi_api_prediction_errors_total", reason="model_not_loaded") == before + 1


def test_metrics_requests_are_not_logged(api):
    client, _ = api
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("app")
    logger.addHandler(handler)
    try:
        client.get("/metrics")
        client.get("/live")
    finally:
        logger.removeHandler(handler)
    paths = [json.loads(line).get("path") for line in stream.getvalue().splitlines()]
    assert "/live" in paths and "/metrics" not in paths
