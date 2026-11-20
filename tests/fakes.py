"""Fake model bundles for tests that need no MLflow server and no real model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.bundle import Bundle, feature_schema, reference_frame
from src.features import FEATURES, build_features


class FakeModel:
    """Predicts a constant; can be told to fail or return bad values."""

    def __init__(self, minutes=12.0, features=FEATURES, fail: Exception | None = None):
        self.minutes = minutes
        self.feature_names_in_ = np.array(features, dtype=object)
        self.fail = fail

    def predict(self, X):
        if self.fail is not None:
            raise self.fail
        return np.full(len(X), self.minutes, dtype=float)


def fake_zones() -> pd.DataFrame:
    """Zone table for the zones used in tests (others, like 264, get -1 features)."""
    zones = pd.DataFrame({
        "LocationID": [74, 132, 161, 162, 236],
        "lat": [40.80, 40.64, 40.76, 40.76, 40.78],
        "lon": [-73.94, -73.79, -73.98, -73.97, -73.95],
        "x": [1000.0, 40000.0, 990.0, 992.0, 997.0],
        "y": [230.0, 170.0, 215.0, 214.0, 222.0],
    })
    return zones.set_index("LocationID")


def good_schema() -> dict:
    return feature_schema(build_features(reference_frame(), fake_zones()))


def make_bundle(version="1", model=None, zone_xy="default", schema="default", warnings=()) -> Bundle:
    return Bundle(
        model=model if model is not None else FakeModel(),
        zone_xy=fake_zones() if isinstance(zone_xy, str) else zone_xy,
        schema=good_schema() if isinstance(schema, str) else schema,
        version=str(version), run_id=f"run-{version}", warnings=tuple(warnings),
    )


class FakeMlflowClient:
    """Stands in for mlflow.MlflowClient: registered versions -> run ids and params."""

    def __init__(self, versions: dict[str, dict], aliases: dict[str, str] | None = None):
        self.versions = versions          # version -> run params
        self.aliases = aliases or {}

    def get_model_version(self, name, version):
        if str(version) not in self.versions:
            raise LookupError(f"RESOURCE_DOES_NOT_EXIST: {name} version {version}")
        return SimpleNamespace(version=str(version), run_id=f"run-{version}")

    def get_model_version_by_alias(self, name, alias):
        return self.get_model_version(name, self.aliases[alias])

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(params=self.versions[run_id.removeprefix("run-")]))
