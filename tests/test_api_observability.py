"""/live, /ready, request IDs, the JSON error format and JSON logs. Fake bundle only (runs in CI)."""

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.observability import JsonFormatter
from tests.fakes import make_bundle

TRIP = {"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    bundle = make_bundle("7")
    monkeypatch.setattr(main, "load_champion", lambda: bundle)
    with TestClient(main.app) as c:
        c.bundle = bundle
        yield c


def assert_error_format(response, status, code):
    assert response.status_code == status
    body = response.json()
    assert body["code"] == code
    assert isinstance(body["message"], str) and body["message"]
    assert body["request_id"] == response.headers["X-Request-ID"]
    return body


def test_live_and_ready(client):
    assert client.get("/live").json() == {"status": "alive"}
    assert client.get("/ready").json() == {"status": "ready", "model_version": "7"}


@pytest.mark.parametrize("method, path, body", [
    ("get", "/live", None), ("get", "/ready", None), ("get", "/health", None),
    ("post", "/predict", TRIP), ("get", "/no-such-page", None),
])
def test_every_response_gets_a_new_request_id(client, method, path, body):
    r = getattr(client, method)(path, **({"json": body} if body else {}))
    rid = r.headers["X-Request-ID"]
    assert len(rid) == 32 and int(rid, 16) >= 0  # uuid4 hex


def test_incoming_request_id_is_returned(client):
    r = client.post("/predict", json=TRIP, headers={"X-Request-ID": "client-req.42"})
    assert r.headers["X-Request-ID"] == "client-req.42"


@pytest.mark.parametrize("bad_id", ["has spaces", "x" * 200, "semi;colon"])
def test_unsafe_request_id_is_replaced(client, bad_id):
    r = client.get("/live", headers={"X-Request-ID": bad_id})
    assert r.headers["X-Request-ID"] != bad_id and len(r.headers["X-Request-ID"]) == 32


def test_prediction_reports_the_model_version(client):
    r = client.post("/predict", json=TRIP)
    assert r.json()["model_version"] == "7" and r.headers["X-Model-Version"] == "7"


def test_error_format_not_found(client):
    assert_error_format(client.get("/no-such-page"), 404, "not_found")


def test_error_format_validation(client):
    body = assert_error_format(client.post("/predict", json={**TRIP, "PULocationID": 0}), 422, "invalid_request")
    assert body["details"][0]["loc"] == ["body", "PULocationID"]


def test_error_format_unauthorized(client):
    assert_error_format(client.post("/reload"), 401, "unauthorized")


def test_error_format_unhandled_error(client):
    client.bundle.model.fail = RuntimeError("boom")  # fails at prediction time, after validation
    body = assert_error_format(client.post("/predict", json=TRIP), 500, "internal_error")
    assert "boom" not in body["message"]  # internal details are logged, not returned


def test_error_format_not_ready(monkeypatch):
    def no_model():
        raise main.bundle_lib.BundleLoadError("registry unreachable")

    monkeypatch.setattr(main, "load_champion", no_model)
    with TestClient(main.app) as c:
        assert_error_format(c.get("/ready"), 503, "not_ready")
        assert_error_format(c.post("/predict", json=TRIP), 503, "model_not_loaded")


def test_request_log_is_json_with_request_id_and_model_version(client):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("app")
    logger.addHandler(handler)
    try:
        client.post("/predict", json=TRIP, headers={"X-Request-ID": "log-test-1"})
    finally:
        logger.removeHandler(handler)
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    finished = [l for l in lines if l["message"] == "request finished"]
    assert finished, lines
    line = finished[-1]
    assert line["request_id"] == "log-test-1" and line["model_version"] == "7"
    assert line["status_code"] == 200 and line["path"] == "/predict"
    assert {"timestamp", "level"} <= set(line)
