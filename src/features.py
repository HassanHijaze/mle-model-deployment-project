"""Feature engineering shared by training and the API.

build_zone_xy() needs geopandas and a download; it is only used during training.
The API gets the finished zone table from MLflow (logged with the model), so it
only needs build_features().
"""

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

FEATURES = [
    "PULocationID",
    "DOLocationID",
    "hour",
    "day_of_week",
    "is_holiday",
    "PU_lat",
    "PU_lon",
    "DO_lat",
    "DO_lon",
    "straight_dist",
]


def build_zone_xy(data_dir: Path) -> pd.DataFrame:
    """Center point of each taxi zone (x/y in feet, lat/lon), indexed by LocationID.

    Downloads the TLC zone shapefile into data_dir/taxi_zones if no .shp file is there yet.
    """
    import io
    import zipfile

    import geopandas as gpd
    import requests

    zone_dir = data_dir / "taxi_zones"
    shp_files = list(zone_dir.rglob("*.shp")) if zone_dir.exists() else []
    if not shp_files:
        r = requests.get(ZONES_URL, timeout=60)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(zone_dir)
        shp_files = list(zone_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {zone_dir} after download")

    gdf = gpd.read_file(shp_files[0]).dissolve(by="LocationID")
    centroids = gdf.geometry.centroid  # feet (NYC map projection)
    latlon = centroids.to_crs(epsg=4326)  # latitude/longitude
    return pd.DataFrame({"x": centroids.x, "y": centroids.y, "lat": latlon.y, "lon": latlon.x})


def build_features(df: pd.DataFrame, zone_xy: pd.DataFrame) -> pd.DataFrame:
    """Turn PULocationID, DOLocationID and tpep_pickup_datetime into the model features."""
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    us_holidays = list(holidays.US(years=pickup.dt.year.unique().tolist()).keys())

    X = pd.DataFrame(index=df.index)
    X["PULocationID"] = df["PULocationID"]
    X["DOLocationID"] = df["DOLocationID"]
    X["hour"] = pickup.dt.hour
    X["day_of_week"] = pickup.dt.day_of_week
    X["is_holiday"] = pickup.dt.date.isin(us_holidays).astype(int)

    # Zone centers and straight-line distance (miles); -1 for unknown zones
    for prefix, col in [("PU", "PULocationID"), ("DO", "DOLocationID")]:
        X[f"{prefix}_lat"] = df[col].map(zone_xy["lat"]).fillna(-1)
        X[f"{prefix}_lon"] = df[col].map(zone_xy["lon"]).fillna(-1)
    dx = df["PULocationID"].map(zone_xy["x"]) - df["DOLocationID"].map(zone_xy["x"])
    dy = df["PULocationID"].map(zone_xy["y"]) - df["DOLocationID"].map(zone_xy["y"])
    X["straight_dist"] = (np.sqrt(dx**2 + dy**2) / 5280).fillna(-1)

    return X[FEATURES]
