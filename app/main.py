"""FastAPI app for NYC yellow taxi trip duration prediction."""

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

from src.features import build_features, build_zone_xy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "nyc-taxi-duration-rf"
MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
)

client = MlflowClient()
_lock = threading.Lock()

zone_xy = None        # built at startup (see lifespan)
model = None
model_version = None


def load_zone_xy():
    """Build the zone lookup table, or return None if it fails (e.g. download or permission error)."""
    logger.info("Building zone map from %s ...", ROOT / "data")
    try:
        zones = build_zone_xy(ROOT / "data")
        logger.info("Zone map ready (%d zones)", len(zones))
        return zones
    except Exception as exc:
        logger.error("Zone map unavailable: %s", exc)
        return None


def load_champion():
    """Load the current champion model if it exists."""
    try:
        logger.info("Looking up '%s' alias of %s ...", MODEL_ALIAS, MODEL_NAME)
        version = str(
            client.get_model_version_by_alias(
                MODEL_NAME, MODEL_ALIAS
            ).version
        )
        logger.info("Champion is version %s, loading model ...", version)

        loaded_model = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}/{version}"
        )
        logger.info("Model version %s loaded", version)

        return loaded_model, version

    except Exception as exc:
        logger.warning("Champion model unavailable: %s", exc)
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the zone map and the champion model when the app starts.

    If either fails, the API still starts; /predict returns 503 and /health shows what is missing.
    """
    global model, model_version, zone_xy
    zone_xy = load_zone_xy()
    model, model_version = load_champion()
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

    if current_zones is None:
        raise HTTPException(
            status_code=503,
            detail="Zone map not loaded. Check the API logs and the data/ folder, then call /reload."
        )
    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Register a champion model and call /reload."
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
    """Reload the champion model, and retry the zone map if it failed at startup."""
    global model, model_version, zone_xy

    new_zones = zone_xy if zone_xy is not None else load_zone_xy()
    new_model, new_version = load_champion()   # slow part, outside the lock

    with _lock:
        old = model_version
        zone_xy = new_zones
        if new_model is not None:
            model = new_model
            model_version = new_version
        return {
            "previous_version": old,
            "current_version": model_version,
            "model_loaded": model is not None,
            "zones_loaded": zone_xy is not None,
        }


@app.get("/health")
def health():
    with _lock:
        return {
            "status": "running",
            "model_loaded": model is not None,
            "model_version": model_version,
            "zones_loaded": zone_xy is not None,
        }