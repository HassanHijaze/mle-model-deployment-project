"""Prometheus metrics of the API (exposed at GET /metrics, scraped by the prometheus service).

Requests are labelled with the route TEMPLATE (e.g. "/predict"), never the raw URL, so the
number of series stays small; unknown paths share the label "unmatched".
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter(
    "taxi_api_requests_total",
    "HTTP requests by method, route and status code.",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "taxi_api_request_duration_seconds",
    "HTTP request latency in seconds, by method and route.",
    ["method", "route"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREDICTIONS = Counter(
    "taxi_api_predictions_total",
    "Successful predictions, by the model version that answered.",
    ["model_version"],
)
PREDICTION_ERRORS = Counter(
    "taxi_api_prediction_errors_total",
    "Failed /predict calls: model_not_loaded (503) or error (500).",
    ["reason"],
)
MODEL_INFO = Gauge(
    "taxi_api_model_info",
    "1 for the model version currently served (no series when no model is loaded).",
    ["model_version"],
)
MODEL_LOADED = Gauge("taxi_api_model_loaded", "1 if a validated model is loaded, else 0.")
MODEL_LOAD_DURATION = Histogram(
    "taxi_api_model_load_duration_seconds",
    "Time to load and validate the champion bundle, by source (startup/reload) and outcome.",
    ["source", "outcome"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)
RELOAD_ATTEMPTS = Counter("taxi_api_reload_attempts_total", "Authenticated /reload calls.")
RELOAD_FAILURES = Counter(
    "taxi_api_reload_failures_total",
    "Failed /reload calls by reason: in_progress, load_failed, invalid_bundle or error.",
    ["reason"],
)

# Pre-create the label sets that are known up front, so they appear as 0 before the first event
for _reason in ("model_not_loaded", "error"):
    PREDICTION_ERRORS.labels(reason=_reason)
for _reason in ("in_progress", "load_failed", "invalid_bundle", "error"):
    RELOAD_FAILURES.labels(reason=_reason)


def set_active_model(version: str | None) -> None:
    """Point the model gauges at the version now served (None = no model)."""
    MODEL_INFO.clear()
    if version is None:
        MODEL_LOADED.set(0)
    else:
        MODEL_INFO.labels(model_version=version).set(1)
        MODEL_LOADED.set(1)


def render() -> tuple[bytes, str]:
    """The current metrics in the Prometheus text format, and its content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
