"""One-command pipeline for the NYC taxi trip-duration project.

    uv run nyc-taxi all       # data -> tune -> train -> select, stops on the first error

Subcommands:
    data     download January and February 2025 into data/ and build the zone table
    tune     src.tuning: Optuna search on the 500k sample, MLflow run "optuna-tuning"
    train    src.train: baseline runs (overall median, zone-pair median, rf-baseline - registered)
             scored on validation and test, + tuned 200-tree test score (rf-final-jan1-25)
    select   src.serving_selection run: candidates, benchmarks, selection rule, February
             evaluation, registers the selected model WITHOUT an alias
    all      data, tune, train and select in that order
    verify   check one registered version in a fresh process (model, feature schema, zone
             table, test predictions) and print OK or the exact problem

train and select use the best params of the latest "optuna-tuning" run in MLflow
(fallback: BEST_PARAMS in src/train.py, with a printed note).

The MLflow address comes from MLFLOW_TRACKING_URI (default http://localhost:5002).
The champion alias is never moved: set it by hand in the MLflow UI.

Expected runtime: data ~2-3 min (first download), tune ~15-40 min, train ~15-30 min,
select ~1-1.5 h; peak RAM 6-8 GB.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable

MLFLOW_START_HINT = "start it with: docker compose up -d postgres mlflow"


class PipelineError(RuntimeError):
    """A step cannot start (for example the MLflow server is not reachable)."""


def _model_version(value: str) -> str:
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"expected a registered version number (1, 2, ...), got {value!r}")
    return str(int(value))


def run_verify(version: str) -> int:
    """Run src.verify in a fresh Python process (only the installed code and the bundle)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    return subprocess.run([sys.executable, "-m", "src.verify", version], cwd=root, env=env).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nyc-taxi", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    sub.add_parser("data", help="download January and February 2025 and build the zone table")
    sub.add_parser("tune", help="Optuna search on the 500k sample, logged as MLflow run optuna-tuning")
    sub.add_parser("train", help="baselines (median, zone-pair median, RF) + tuned 200-tree test score, logged to MLflow")
    sub.add_parser("select", help="choose, evaluate on February and register the serving model (no alias)")
    sub.add_parser("all", help="data -> tune -> train -> select, stopping on the first error")
    v = sub.add_parser("verify", help="check one registered model version before promoting it")
    v.add_argument("version", type=_model_version, help="registered version number of nyc-taxi-duration-rf")
    return parser


# ---------------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------------


def check_mlflow() -> str:
    """Stop with a clear message if the MLflow server is not reachable. Returns its URI."""
    from src.experiment import TRACKING_URI

    if not TRACKING_URI.startswith(("http://", "https://")):
        return TRACKING_URI  # local file/database store: nothing to reach
    try:
        with urllib.request.urlopen(f"{TRACKING_URI.rstrip('/')}/health", timeout=5) as response:
            if response.status != 200:
                raise PipelineError(f"MLflow server at {TRACKING_URI} answered HTTP {response.status}; "
                                    f"{MLFLOW_START_HINT}")
    except (urllib.error.URLError, OSError) as exc:
        raise PipelineError(f"MLflow server not reachable at {TRACKING_URI} ({exc}); "
                            f"{MLFLOW_START_HINT}") from None
    return TRACKING_URI


def run_data() -> dict:
    from src.features import build_zone_xy
    from src.serving_selection import DATA_DIR, describe_period, load_month
    from src.train import DATA_PATH, download_january, load_january

    if not DATA_PATH.exists():
        download_january()
    print("January: ", describe_period(load_january()))
    print("February:", describe_period(load_month(2025, 2)))  # downloads on first use
    zones = build_zone_xy(DATA_DIR)  # downloads the TLC zone shapefile on first use
    print(f"Zone table: {len(zones)} zones")
    return {}


def run_tune() -> dict:
    check_mlflow()
    from src import tuning

    return tuning.main()


def run_train() -> dict:
    check_mlflow()
    from src import train

    return train.main()


def run_select() -> dict:
    check_mlflow()
    from src.serving_selection import run_selection

    return run_selection()


STEPS: dict[str, Callable[[], dict]] = {
    "data": run_data, "tune": run_tune, "train": run_train, "select": run_select}  # "all" order


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def run_step(name: str) -> dict:
    """Run one step between clear start/end lines with the elapsed time."""
    print(f"\n==> [{name}] start  ({time.strftime('%H:%M:%S')})", flush=True)
    t0 = time.perf_counter()
    try:
        result = STEPS[name]() or {}
    except BaseException:
        print(f"<== [{name}] FAILED after {_elapsed(time.perf_counter() - t0)}", flush=True)
        raise
    print(f"<== [{name}] done in {_elapsed(time.perf_counter() - t0)}", flush=True)
    return result


def print_next_steps(results: dict) -> None:
    """Registered versions, the (unchanged) champion and how to promote a version."""
    from src.serving_selection import alias_version
    from src.train import MODEL_NAME

    print(f"\nRegistered versions of {MODEL_NAME} (no alias set):")
    if "baseline_version" in results:
        print(f"  rf-baseline:                 version {results['baseline_version']}")
    if "registered_version" in results:
        print(f"  serving model ({results.get('selected', '?')}): version {results['registered_version']}")
    print(f"  champion alias (unchanged):  {alias_version(MODEL_NAME, 'champion') or 'not set'}")
    print("\nNext steps (docs/operations.md):")
    print("  1. uv run nyc-taxi verify <version>")
    print("  2. set the 'champion' alias to that version in the MLflow UI")
    print('  3. curl -X POST http://localhost:8000/reload -H "X-Admin-Token: <your token>"')


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        try:
            check_mlflow()
        except PipelineError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return run_verify(args.version)

    names = list(STEPS) if args.command == "all" else [args.command]
    t0 = time.perf_counter()
    results: dict = {}
    try:
        if any(n in ("tune", "train", "select") for n in names):
            check_mlflow()  # fail before the (slow) data step, not after it
        for name in names:
            results.update(run_step(name))
    except PipelineError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if args.command == "all":
        print(f"\nPipeline finished in {_elapsed(time.perf_counter() - t0)}")
    if args.command in ("train", "select", "all"):  # the commands that register models
        print_next_steps(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
