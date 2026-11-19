"""FastAPI app for NYC yellow taxi trip duration prediction.

At startup the API loads the "champion" model from MLflow, together with the zone
table (zone_xy.csv) logged in the same run. No download or geopandas is needed.
"""

import os
import threading
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mlflow import MlflowClient
from pydantic import BaseModel, Field, field_validator

from src.features import build_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "nyc-taxi-duration-rf"
MODEL_ALIAS = "champion"
ZONE_FILE = "zone_xy.csv"

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
)

client = MlflowClient()
_lock = threading.Lock()          # protects model, model_version and zone_xy
_reload_lock = threading.Lock()   # only one /reload at a time (the model can be several GB)

zone_xy = None
model = None
model_version = None


def load_zones(run_id):
    """Load the zone table logged with the model's run, or None if it is missing."""
    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ZONE_FILE)
        zones = pd.read_csv(path, index_col="LocationID")
        logger.info("Zone map loaded from MLflow (%d zones)", len(zones))
        return zones
    except Exception as exc:
        logger.error("Zone map unavailable for run %s: %s", run_id, exc)
        return None


def load_champion():
    """Load the champion model and its zone table. Returns (model, version, zones)."""
    try:
        logger.info("Looking up '%s' alias of %s ...", MODEL_ALIAS, MODEL_NAME)
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        version = str(mv.version)
        logger.info("Champion is version %s, loading model ...", version)

        loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
        logger.info("Model version %s loaded", version)

        return loaded_model, version, load_zones(mv.run_id)

    except Exception as exc:
        logger.warning("Champion model unavailable: %s", exc)
        return None, None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model and its zone map when the app starts.

    If either fails, the API still starts; /predict returns 503 and /health shows what is missing.
    """
    global model, model_version, zone_xy
    model, model_version, zone_xy = load_champion()
    logger.info("Startup finished (zones_loaded=%s, model_loaded=%s)",
                zone_xy is not None, model is not None)
    yield


app = FastAPI(title="NYC Taxi Trip Duration", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        Path(__file__).parent / "static" / "index.html"
    )


class Trip(BaseModel):
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    pickup_datetime: datetime

    @field_validator("pickup_datetime")
    @classmethod
    def no_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError(
                "pickup_datetime must be local New York time "
                "without a timezone"
            )
        return value


class Prediction(BaseModel):
    duration_minutes: float
    model_version: str


@app.post("/predict")
def predict(trip: Trip) -> Prediction:
    with _lock:
        current_model = model
        current_version = model_version
        current_zones = zone_xy

    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Register a champion model and call /reload."
        )
    if current_zones is None:
        raise HTTPException(
            status_code=503,
            detail="Zone map not loaded. Register the model with its zone table (zone_xy.csv) and call /reload."
        )

    df = pd.DataFrame([{
        "PULocationID": trip.PULocationID,
        "DOLocationID": trip.DOLocationID,
        "tpep_pickup_datetime": trip.pickup_datetime,
    }])

    X = build_features(df, current_zones)

    return Prediction(
        duration_minutes=round(
            float(current_model.predict(X)[0]), 2
        ),
        model_version=current_version
    )


@app.post("/reload")
def reload() -> dict:
    """Reload the champion model and its zone map.

    Only one reload runs at a time; a second call while one is running gets 409.
    """
    global model, model_version, zone_xy

    if not _reload_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A reload is already running. Try again shortly.")

    try:
        new_model, new_version, new_zones = load_champion()   # slow part, outside _lock

        with _lock:
            old = model_version
            if new_model is not None:
                model, model_version, zone_xy = new_model, new_version, new_zones
            return {
                "previous_version": old,
                "current_version": model_version,
                "model_loaded": model is not None,
                "zones_loaded": zone_xy is not None,
            }
    finally:
        _reload_lock.release()


@app.get("/health")
def health():
    with _lock:
        return {
            "status": "running",
            "model_loaded": model is not None,
            "model_version": model_version,
            "zones_loaded": zone_xy is not None,
        }