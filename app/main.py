
"""FastAPI app for NYC yellow taxi trip duration prediction."""

import os
import threading
import logging
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
zone_xy = build_zone_xy(ROOT / "data")
_lock = threading.Lock()

app = FastAPI(title="NYC Taxi Trip Duration")

model = None
model_version = None


def load_champion():
    """Load the current champion model if it exists."""
    try:
        version = str(
            client.get_model_version_by_alias(
                MODEL_NAME, MODEL_ALIAS
            ).version
        )

        loaded_model = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}/{version}"
        )

        return loaded_model, version

    except Exception as exc:
        logger.warning("Champion model unavailable: %s", exc)
        return None, None


@app.on_event("startup")
def startup():
    global model, model_version
    model, model_version = load_champion()


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

    X = build_features(df, zone_xy)

    return Prediction(
        duration_minutes=round(
            float(current_model.predict(X)[0]), 2
        ),
        model_version=current_version
    )


@app.post("/reload")
def reload() -> dict:
    global model, model_version

    new_model, new_version = load_champion()

    with _lock:
        old = model_version

        if new_model is not None:
            model = new_model
            model_version = new_version

    return {
        "previous_version": old,
        "current_version": model_version,
        "model_loaded": model is not None
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "model_version": model_version
    }
