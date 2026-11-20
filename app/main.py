"""FastAPI app for NYC yellow taxi trip duration prediction.

At startup the API loads the "champion" model from MLflow as ONE bundle: the model,
its zone table (zone_xy.csv) and its feature schema (feature_schema.json), all logged
in the same run (src/bundle.py). The bundle is validated before it is served.

Endpoints:
    POST /predict   trip duration; the response names the model version that answered
    POST /reload    load the current champion as a new bundle, validate it, then swap
                    (needs the X-Admin-Token header; one reload at a time). If loading or
                    validation fails, the old model keeps serving and a JSON error says why.
    GET  /live      the process is alive
    GET  /ready     a validated model is loaded (503 otherwise)
    GET  /health    model, zone table and last reload status
    GET  /metrics   Prometheus metrics (app/metrics.py); not written to the request log

Every response has an X-Request-ID header; errors use one JSON format (app/observability.py).
The API never moves the champion alias.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import metrics
from app.observability import REQUEST_ID_HEADER, error_body, request_id_var, resolve_request_id, setup_logging
from src import bundle as bundle_lib
from src.experiment import MODEL_NAME
from src.features import build_features

setup_logging()
logger = logging.getLogger("app")

MODEL_ALIAS = "champion"
ADMIN_TOKEN_ENV = "ADMIN_TOKEN"
MODEL_VERSION_HEADER = "X-Model-Version"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002"))

_state_lock = threading.Lock()  # guards _bundle and _last_reload (swaps are a single assignment)
_reload_lock = threading.Lock()  # only one reload at a time (loading and validating a model takes a while)
_bundle: bundle_lib.Bundle | None = None
_last_reload: dict = {"at": None, "source": None, "ok": None, "version": None, "error": None}


# ---------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------


class ApiError(Exception):
    """An error returned to the client in the API's JSON error format."""

    def __init__(self, status_code: int, code: str, message: str, **extra):
        super().__init__(message)
        self.status_code, self.code, self.message, self.extra = status_code, code, message, extra


class InvalidBundleError(Exception):
    """The bundle loaded but failed validation."""

    def __init__(self, version: str, problems: list[str]):
        super().__init__("; ".join(problems))
        self.version, self.problems = version, problems


HTTP_ERROR_CODES = {404: "not_found", 405: "method_not_allowed"}


# ---------------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------------


def load_champion() -> bundle_lib.Bundle:
    """Fetch the champion bundle from MLflow (raises bundle_lib.BundleLoadError)."""
    return bundle_lib.fetch_bundle(MODEL_NAME, alias=MODEL_ALIAS)


def load_and_validate(source: str = "reload") -> bundle_lib.Bundle:
    """Fetch the champion bundle and validate it. Raises BundleLoadError or InvalidBundleError.

    The duration is recorded in taxi_api_model_load_duration_seconds{source, outcome}.
    """
    start = time.perf_counter()
    outcome = "error"
    try:
        new = load_champion()
        problems = bundle_lib.validate_bundle(new)
        if problems:
            outcome = "invalid_bundle"
            raise InvalidBundleError(new.version, problems)
        outcome = "ok"
        return new
    except bundle_lib.BundleLoadError:
        outcome = "load_failed"
        raise
    finally:
        metrics.MODEL_LOAD_DURATION.labels(source=source, outcome=outcome).observe(
            time.perf_counter() - start
        )


def current_bundle() -> bundle_lib.Bundle | None:
    with _state_lock:
        return _bundle


def _record(source: str, ok: bool, version: str | None, error: str | None) -> None:
    with _state_lock:
        _last_reload.update(
            at=datetime.now(UTC).isoformat(timespec="seconds"),
            source=source,
            ok=ok,
            version=version,
            error=error,
        )


def _swap(new: bundle_lib.Bundle | None) -> None:
    global _bundle
    with _state_lock:
        _bundle = new
        metrics.set_active_model(new.version if new is not None else None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and validate the champion bundle at startup.

    If that fails the API still starts: /live is OK, /ready and /predict return 503 and
    /health shows the reason; an authenticated /reload can load a model later.
    """
    try:
        new = load_and_validate(source="startup")
    except bundle_lib.BundleLoadError as exc:
        logger.warning("No champion model at startup", extra={"event": "startup", "reason": str(exc)})
        _record("startup", False, None, str(exc))
        _swap(None)
    except InvalidBundleError as exc:
        logger.error(
            "Champion bundle failed validation at startup",
            extra={"event": "startup", "model_version": exc.version, "reason": str(exc)},
        )
        _record("startup", False, exc.version, f"version {exc.version} failed validation: {exc}")
        _swap(None)
    else:
        _swap(new)
        _record("startup", True, new.version, None)
        for warning in new.warnings:
            logger.warning(warning, extra={"model_version": new.version})
        logger.info("Champion bundle loaded", extra={"event": "startup", "model_version": new.version})
    yield


app = FastAPI(title="NYC Taxi Trip Duration", lifespan=lifespan)


# ---------------------------------------------------------------------------------
# Request IDs, request log and error format
# ---------------------------------------------------------------------------------


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None) or request_id_var.get()


@app.middleware("http")
async def request_context(request: Request, call_next):
    """X-Request-ID on every response, request metrics, one JSON log line per request, JSON 500s."""
    request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error", extra={"event": "unhandled_error"})
            response = JSONResponse(
                status_code=500, content=error_body("internal_error", "Internal server error.", request_id)
            )
        response.headers[REQUEST_ID_HEADER] = request_id
        duration = time.perf_counter() - start
        route = getattr(request.scope.get("route"), "path", None) or "unmatched"  # template, not raw URL
        metrics.REQUESTS.labels(method=request.method, route=route, status=str(response.status_code)).inc()
        metrics.REQUEST_LATENCY.labels(method=request.method, route=route).observe(duration)
        if route != "/metrics":  # scraped every few seconds: kept out of the request log
            logger.info(
                "request finished",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "model_version": getattr(request.state, "model_version", None),
                },
            )
        return response
    finally:
        request_id_var.reset(token)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    logger.warning(exc.message, extra={"event": exc.code})
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.code, exc.message, _request_id(request), **exc.extra),
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    code = HTTP_ERROR_CODES.get(exc.status_code, f"http_{exc.status_code}")
    return JSONResponse(
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
        content=error_body(code, str(exc.detail), _request_id(request)),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = [
        {"loc": list(e.get("loc", ())), "message": e.get("msg"), "type": e.get("type")} for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_body(
            "invalid_request", "The request is invalid; see details.", _request_id(request), details=details
        ),
    )


# ---------------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


class Trip(BaseModel):
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    pickup_datetime: datetime

    @field_validator("pickup_datetime")
    @classmethod
    def no_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError("pickup_datetime must be local New York time without a timezone")
        return value


class Prediction(BaseModel):
    duration_minutes: float
    model_version: str


@app.post("/predict")
def predict(trip: Trip, request: Request, response: Response) -> Prediction:
    current = current_bundle()  # one reference: a reload swapping meanwhile does not affect this request
    if current is None:
        metrics.PREDICTION_ERRORS.labels(reason="model_not_loaded").inc()
        raise ApiError(
            503, "model_not_loaded", "No model loaded. Register a champion model and call /reload."
        )

    df = pd.DataFrame(
        [
            {
                "PULocationID": trip.PULocationID,
                "DOLocationID": trip.DOLocationID,
                "tpep_pickup_datetime": trip.pickup_datetime,
            }
        ]
    )
    try:
        minutes = float(current.model.predict(build_features(df, current.zone_xy))[0])
    except Exception:
        metrics.PREDICTION_ERRORS.labels(reason="error").inc()
        raise  # answered as a JSON 500 by the request middleware

    metrics.PREDICTIONS.labels(model_version=current.version).inc()
    request.state.model_version = current.version
    response.headers[MODEL_VERSION_HEADER] = current.version
    return Prediction(duration_minutes=round(minutes, 2), model_version=current.version)


def require_admin(token: str | None) -> None:
    expected = os.getenv(ADMIN_TOKEN_ENV)
    if not expected:
        raise ApiError(
            503, "reload_disabled", f"{ADMIN_TOKEN_ENV} is not set on the server, so /reload is disabled."
        )
    if not token or not secrets.compare_digest(token.encode(), expected.encode()):
        raise ApiError(401, "unauthorized", "Missing or wrong X-Admin-Token header.")


@app.post("/reload")
def reload(x_admin_token: str | None = Header(default=None)) -> dict:
    """Load the champion as a new bundle, validate it, and only then swap it in.

    One reload at a time (409 otherwise). On failure the old model keeps serving:
    502 reload_failed (registry/model could not be loaded) or 422 bundle_invalid
    (loaded but failed validation, with the list of problems).
    """
    require_admin(x_admin_token)
    metrics.RELOAD_ATTEMPTS.inc()
    if not _reload_lock.acquire(blocking=False):
        metrics.RELOAD_FAILURES.labels(reason="in_progress").inc()
        raise ApiError(409, "reload_in_progress", "A reload is already running. Try again shortly.")
    try:
        previous = current_bundle()
        serving = previous.version if previous else None
        try:
            new = load_and_validate(source="reload")  # slow part: requests keep using the old bundle
        except bundle_lib.BundleLoadError as exc:
            metrics.RELOAD_FAILURES.labels(reason="load_failed").inc()
            _record("reload", False, None, str(exc))
            raise ApiError(
                502, "reload_failed", f"Could not load the champion bundle: {exc}", serving_version=serving
            ) from exc
        except InvalidBundleError as exc:
            metrics.RELOAD_FAILURES.labels(reason="invalid_bundle").inc()
            _record("reload", False, exc.version, f"version {exc.version} failed validation: {exc}")
            raise ApiError(
                422,
                "bundle_invalid",
                f"Version {exc.version} failed validation; still serving version {serving}.",
                rejected_version=exc.version,
                problems=exc.problems,
                serving_version=serving,
            ) from exc
        except Exception as exc:  # e.g. a corrupted model that crashes during validation
            metrics.RELOAD_FAILURES.labels(reason="error").inc()
            _record("reload", False, None, f"{type(exc).__name__}: {exc}")
            raise ApiError(
                500, "reload_failed", f"Reload failed: {type(exc).__name__}: {exc}", serving_version=serving
            ) from exc

        _swap(new)
        _record("reload", True, new.version, None)
        logger.info(
            "Model reloaded",
            extra={"event": "reload", "model_version": new.version, "previous_version": serving},
        )
        return {
            "status": "reloaded",
            "previous_version": serving,
            "current_version": new.version,
            "model_loaded": True,
            "zones_loaded": new.zone_xy is not None,
            "warnings": list(new.warnings),
        }
    finally:
        _reload_lock.release()


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus metrics in the text exposition format (scraped by the prometheus service)."""
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.get("/live")
def live() -> dict:
    """The process is up and answering (no model needed)."""
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict:
    """A validated model bundle is loaded and can answer /predict."""
    current = current_bundle()
    if current is None:
        with _state_lock:
            reason = _last_reload.get("error") or "no model loaded"
        raise ApiError(503, "not_ready", f"No validated model loaded: {reason}")
    return {"status": "ready", "model_version": current.version}


@app.get("/health")
def health() -> dict:
    current = current_bundle()
    with _state_lock:
        last = dict(_last_reload)
    return {
        "status": "running",
        "model_loaded": current is not None,
        "model_version": current.version if current else None,
        "zones_loaded": current is not None and current.zone_xy is not None,
        "schema_logged": current is not None and current.schema is not None,
        "last_reload": last,
    }
