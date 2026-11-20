"""Load test of the prediction API with Locust.

Each simulated user sends /predict requests for random valid trips (zones 1-263, pickup
times in January 2025, New York local time) and, about 1 call in 20, checks /ready.

Run all three levels and save CSV results:   uv run python tests/load/run_load_test.py
Interactive UI (http://localhost:8089):      uv run locust -f tests/load/locustfile.py --host http://localhost:8000
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from locust import HttpUser, between, task

ZONES = range(1, 264)  # TLC zones with a shape (264/265 are "Unknown"/"Outside NYC")
JANUARY_START = datetime(2025, 1, 1)
JANUARY_MINUTES = 31 * 24 * 60


def random_trip() -> dict:
    pickup = JANUARY_START + timedelta(minutes=random.randrange(JANUARY_MINUTES))
    return {
        "PULocationID": random.choice(ZONES),
        "DOLocationID": random.choice(ZONES),
        "pickup_datetime": pickup.isoformat(timespec="seconds"),
    }


class TaxiUser(HttpUser):
    wait_time = between(0.5, 1.5)  # think time between a user's requests (seconds)

    @task(19)
    def predict(self) -> None:
        with self.client.post("/predict", json=random_trip(), name="/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
            elif "model_version" not in r.json():
                r.failure("response without model_version")

    @task(1)
    def ready(self) -> None:
        with self.client.get("/ready", name="/ready", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
