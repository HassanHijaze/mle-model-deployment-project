"""Verify one registered model version before promoting it.

    uv run nyc-taxi verify <version>        (runs this module in a fresh process)
    uv run python -m src.verify <version>

Loads the version with its zone table and feature schema (src.bundle.fetch_bundle),
runs the same checks as the API's /reload (src.bundle.validate_bundle), predicts the
reference trips and prints "OK" or the exact problem. Exit code 0 = OK, 1 = problem.
Never touches the champion alias.
"""

from __future__ import annotations

import sys

from src.bundle import (REFERENCE_TRIPS, BundleLoadError, fetch_bundle, predict_reference_trips,
                        validate_bundle, version_differences)

MODEL_NAME = "nyc-taxi-duration-rf"


def verify_version(version: str | int, model_name: str = MODEL_NAME) -> int:
    """Print the verification report for one version. Returns 0 (OK) or 1 (problem)."""
    print(f"Verifying {model_name} version {version} ...")
    try:
        bundle = fetch_bundle(model_name, version=version)
    except BundleLoadError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"  model loaded (run {bundle.run_id})")
    start, end = bundle.params.get("bundle_training_start"), bundle.params.get("bundle_training_end")
    print(f"  training period: {f'{start} .. {end}' if start else 'not logged'}")
    if bundle.zone_xy is not None:
        print(f"  zone table: {len(bundle.zone_xy)} zones")
    if bundle.schema is not None:
        print(f"  feature schema: {len(bundle.schema.get('features', []))} features")
    for warning in (*bundle.warnings, *version_differences(bundle.params)):
        print(f"  warning: {warning}")

    problems = validate_bundle(bundle)
    if problems:
        print("FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    for (pu, do, when), minutes in zip(REFERENCE_TRIPS, predict_reference_trips(bundle.model, bundle.zone_xy)):
        print(f"  {pu:>3} -> {do:<3} {when}: {minutes:.2f} min")
    print("OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    import mlflow

    from src.experiment import TRACKING_URI

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m src.verify <version>", file=sys.stderr)
        return 2
    mlflow.set_tracking_uri(TRACKING_URI)
    return verify_version(args[0])


if __name__ == "__main__":
    sys.exit(main())
