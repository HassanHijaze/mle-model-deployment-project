"""Serving-model selection: candidate training, fresh-process benchmarks, the selection
rule, the February holdout and registration of the model to serve.

All training happens in the "run" command (also `nyc-taxi select`);
notebooks/model_serving_selection.ipynb only reads the logged results from MLflow.
The evaluation helpers (metrics, baselines, error segments) are shared with src.train
and notebooks/exploration.ipynb.

Commands (from the project root, MLflow server running):

    uv run python -m src.serving_selection run
        1. trains the candidates on Jan 1-20 and scores them on validation (Jan 21-25);
           they are built from the best params of the latest "optuna-tuning" run
           (fallback: BEST_PARAMS in src/train.py, see src.train.load_tuned_params)
        2. benchmarks every candidate in a fresh Python process
        3. applies the selection rule (SelectionPolicy)
        4. refits the selected configuration once from scratch (repeatability check)
        5. trains it on all of January and evaluates it once on February 2025
        6. registers it in MLflow WITHOUT an alias (the champion alias is never moved)
        Logged as one parent run "serving-selection" (comparison table CSV, two charts
        with the baseline runs of src.train as reference lines,
        selection rule text, "selected_candidate" tag, February metrics, the serving
        model and its zone table) with one child run per candidate (metrics + status).
        Runtime roughly 1-1.5 h, peak RAM 6-8 GB, ~6 GB free disk in artifacts/.

    uv run python -m src.serving_selection measure --model m.skops --bench bench.parquet
        benchmarks one model in this process and prints one JSON line (used by "run").
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import zone_pair_median_predict  # February comparison (see holdout_metrics)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "artifacts" / "serving_selection"  # git-ignored (large model files)

# Same skops settings the project uses when logging models to MLflow (src/experiment.py)
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]
RANDOM_STATE = 42

# MLflow names (read back by the notebooks)
PARENT_RUN_NAME = "serving-selection"
TABLE_FILE = "candidates_ranked.csv"
SIZE_CHART_FILE = "accuracy_vs_size.png"
LATENCY_CHART_FILE = "accuracy_vs_latency.png"
RULE_FILE = "selection_rule.txt"
TABLE_COLUMNS = ["candidate", "status", "val_rmse", "rmse_vs_best_pct", "val_mae", "size_mb",
                 "peak_ram_mb", "load_s", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                 "single_throughput_rps", "batch_1000_ms", "batch_throughput_rows_s"]
BENCHMARK_COLUMNS = ["size_mb", "peak_ram_mb", "load_s", "latency_p50_ms", "latency_p95_ms",
                     "latency_p99_ms", "single_throughput_rps", "batch_1000_ms",
                     "batch_throughput_rows_s"]

# ---------------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------------

TREE_COUNTS = (25, 50, 75, 100, 150, 200)

# Compact variants: tuned settings with one change that makes each tree smaller.
# Each variant is trained once with 50 trees; the 25-tree candidate is its first 25 trees.
VARIANT_CHANGES = {
    "leaf20": {"min_samples_leaf": 20},
    "leaf50": {"min_samples_leaf": 50},
    "depth18": {"max_depth": 18},
    "depth15": {"max_depth": 15},
}
VARIANT_TREE_COUNTS = (25, 50)


@dataclass(frozen=True)
class Family:
    """One Random Forest fit; its candidates are the first n trees of that forest."""

    name: str
    params: dict
    tree_counts: tuple[int, ...]

    def candidate_name(self, n_trees: int) -> str:
        return f"{self.name}-{n_trees}"


def candidate_families(tuned_params: dict) -> list[Family]:
    """The tuned settings with 25-200 trees, plus the compact variants."""
    tuned = {k: v for k, v in tuned_params.items() if k != "n_estimators"}
    families = [Family("tuned", tuned, TREE_COUNTS)]
    for name, change in VARIANT_CHANGES.items():
        families.append(Family(name, {**tuned, **change}, VARIANT_TREE_COUNTS))
    return families


# ---------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------

MONTH_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]


def load_month(year: int, month: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Clean trips of one month (pickup inside the month, 1-60 minutes), sorted by pickup.

    Same cleaning as src.train.clean_trips (which is fixed to January). The parquet file
    is downloaded once into data_dir (only the columns used).
    """
    path = data_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        pd.read_parquet(MONTH_URL.format(year=year, month=month), columns=COLUMNS).to_parquet(tmp)
        tmp.replace(path)

    df = pd.read_parquet(path, columns=COLUMNS)
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    pickup = df["tpep_pickup_datetime"]
    df = df[(pickup >= start) & (pickup < end)].sort_values("tpep_pickup_datetime")
    df = df.reset_index(drop=True)
    df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    return df[df["duration"].between(1, 60)]


def describe_period(df: pd.DataFrame) -> str:
    """'2025-01-01 .. 2025-01-20 (2,061,234 trips)' for a frame of trips."""
    pickup = df["tpep_pickup_datetime"]
    return f"{pickup.min():%Y-%m-%d} .. {pickup.max():%Y-%m-%d} ({len(df):,} trips)"


# ---------------------------------------------------------------------------------
# Metrics, baselines and error segments
# ---------------------------------------------------------------------------------

AIRPORT_ZONES = (1, 132, 138)  # Newark (EWR), JFK, LaGuardia
SHORT_TRIP_MAX_MIN = 5.0  # actual duration < 5 min
LONG_TRIP_MIN_MIN = 30.0  # actual duration > 30 min
UNCOMMON_PAIR_MAX_TRIPS = 10  # zone pair seen fewer than 10 times in the training period
SEGMENTS = ("short", "long", "airport", "uncommon_pair")  # keys of segment_masks()


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """RMSE and MAE in minutes."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}mae": float(np.mean(np.abs(err))),
    }


def zone_pair_counts(train_df: pd.DataFrame) -> pd.Series:
    """Number of training trips per (PULocationID, DOLocationID) pair."""
    return train_df.groupby(["PULocationID", "DOLocationID"]).size()


def segment_masks(eval_df: pd.DataFrame, pair_counts: pd.Series) -> dict[str, np.ndarray]:
    """Boolean masks for the error segments.

    short: actual duration < 5 min; long: actual duration > 30 min;
    airport: pickup or drop-off at EWR/JFK/LGA;
    uncommon_pair: zone pair with fewer than 10 trips in the training period (incl. unseen).
    """
    keys = pd.MultiIndex.from_frame(eval_df[["PULocationID", "DOLocationID"]])
    seen = pair_counts.reindex(keys).fillna(0).to_numpy()
    duration = eval_df["duration"].to_numpy()
    return {
        "short": duration < SHORT_TRIP_MAX_MIN,
        "long": duration > LONG_TRIP_MIN_MIN,
        "airport": (
            eval_df["PULocationID"].isin(AIRPORT_ZONES) | eval_df["DOLocationID"].isin(AIRPORT_ZONES)
        ).to_numpy(),
        "uncommon_pair": seen < UNCOMMON_PAIR_MAX_TRIPS,
    }


def segment_metrics(eval_df: pd.DataFrame, y_pred, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """RMSE and MAE per segment, e.g. {'short_rmse': ..., 'short_mae': ..., ...}."""
    y = eval_df["duration"].to_numpy()
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {}
    for name, mask in masks.items():
        if mask.any():
            out.update(regression_metrics(y[mask], y_pred[mask], prefix=f"{name}_"))
        else:
            out.update({f"{name}_rmse": float("nan"), f"{name}_mae": float("nan")})
    return out


def segment_sizes(masks: dict[str, np.ndarray]) -> pd.Series:
    """Number of trips in each segment."""
    return pd.Series({name: int(mask.sum()) for name, mask in masks.items()}, name="trips")


# ---------------------------------------------------------------------------------
# Training and forest slicing
# ---------------------------------------------------------------------------------


def fit_rf(params: dict, X: pd.DataFrame, y: pd.Series, n_estimators: int | None = None):
    """Fit a RandomForestRegressor with the project's fixed seed (all CPU cores)."""
    from sklearn.ensemble import RandomForestRegressor

    params = dict(params)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    return RandomForestRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE).fit(X, y)


def slice_forest(forest, n_trees: int):
    """A copy of the forest that keeps only its first n_trees trees.

    With a fixed random_state, sklearn seeds tree i the same way whatever n_estimators is,
    so this equals training with n_estimators=n_trees (checked by check_repeatability()).
    """
    if n_trees > len(forest.estimators_):
        raise ValueError(f"forest has only {len(forest.estimators_)} trees, asked for {n_trees}")
    small = copy.copy(forest)
    small.estimators_ = forest.estimators_[:n_trees]
    small.n_estimators = n_trees
    return small


def predict_tree_counts(forest, X: pd.DataFrame, tree_counts) -> dict[int, np.ndarray]:
    """Predictions of the first n trees for every n in tree_counts, in one pass over the trees."""
    X32 = np.ascontiguousarray(X.to_numpy(dtype=np.float32))  # what the forest passes to its trees
    wanted = sorted(set(tree_counts))
    total = np.zeros(len(X32))
    preds = {}
    for i, tree in enumerate(forest.estimators_[: wanted[-1]], start=1):
        total += tree.predict(X32, check_input=False)
        if i in wanted:
            preds[i] = total / i
    return preds


def node_count(forest) -> int:
    return int(sum(t.tree_.node_count for t in forest.estimators_))


def save_skops(model, path: Path) -> float:
    """Save with skops (the format MLflow uses in this project). Returns the size in MB (MiB)."""
    import skops.io as sio

    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)
    return path.stat().st_size / 2**20


def train_family(
    family: Family,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    val_df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    model_dir: Path = OUTPUT_DIR / "models",
) -> pd.DataFrame:
    """Fit one family, score every tree count on validation and save each candidate as skops.

    Returns one row per candidate (accuracy, segment errors, size, nodes, model path).
    The fitted forest is released before returning, so only one family is in memory at a time.
    """
    n_max = max(family.tree_counts)
    t0 = time.perf_counter()
    forest = fit_rf(family.params, X_train, y_train, n_estimators=n_max)
    fit_s = time.perf_counter() - t0

    rows = []
    for n, pred in predict_tree_counts(forest, X_val, family.tree_counts).items():
        small = slice_forest(forest, n)
        path = model_dir / f"{family.candidate_name(n)}.skops"
        size_mb = save_skops(small, path)
        rows.append({
            "candidate": family.candidate_name(n),
            "family": family.name,
            "n_trees": n,
            **{k: v for k, v in family.params.items()},
            **regression_metrics(val_df["duration"], pred, prefix="val_"),
            **segment_metrics(val_df, pred, masks),
            "size_mb": size_mb,
            "nodes": node_count(small),
            "family_fit_s": fit_s,
            "model_path": str(path),
        })
        del small
    del forest
    return pd.DataFrame(rows)


def check_repeatability(params: dict, n_trees: int, X_train, y_train, X_val, reference_pred) -> dict:
    """Train the configuration once from scratch with exactly n_trees and compare with the
    reference predictions (from the sliced forest that was measured).

    The sliced forest came from a separate fit with more trees, so a match shows both that
    slicing equals training with n_estimators=n_trees and that the fixed seed reproduces the
    model. Returns the largest absolute difference."""
    pred = fit_rf(params, X_train, y_train, n_estimators=n_trees).predict(X_val)
    return {
        "max_abs_diff_fresh_vs_sliced": float(np.max(np.abs(pred - np.asarray(reference_pred)))),
        "identical": bool(np.allclose(pred, reference_pred, rtol=0, atol=1e-9)),
    }


# ---------------------------------------------------------------------------------
# Fresh-process benchmark
# ---------------------------------------------------------------------------------

BENCH_ROWS = 2_000  # validation rows used for the speed benchmark
WARMUP_REQUESTS = 50
TIMED_REQUESTS = 1_000
BATCH_SIZE = 1_000
BATCH_REPEATS = 7


def write_bench_features(X_val: pd.DataFrame, path: Path = OUTPUT_DIR / "bench_features.parquet") -> Path:
    """Fixed random sample of validation feature rows, shared by all candidate benchmarks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    X_val.sample(n=min(BENCH_ROWS, len(X_val)), random_state=RANDOM_STATE).to_parquet(path)
    return path


def _peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (Windows: peak working set)."""
    import psutil

    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KB on Linux


def _current_rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def run_benchmark(model_path: Path, bench_path: Path) -> dict:
    """Load one model and time it. Meant to run in a fresh process (see measure_in_fresh_process).

    Conditions (same for every candidate): model.n_jobs = 1 (one request uses one core, as in
    the API), 1-row pandas DataFrames with the feature names, 50 warm-up requests, then 1,000
    timed sequential requests on different rows; batch = 1,000 rows, median of 7 runs.
    """
    import skops.io as sio
    import sklearn.ensemble  # noqa: F401  (counted in the baseline memory, not in the model)

    X = pd.read_parquet(bench_path)
    rss_before_mb = _current_rss_mb()

    t0 = time.perf_counter()
    model = sio.load(model_path, trusted=SKOPS_TRUSTED_TYPES)
    load_s = time.perf_counter() - t0
    model.n_jobs = 1
    rss_after_load_mb = _current_rss_mb()

    rows = [X.iloc[[i % len(X)]] for i in range(WARMUP_REQUESTS + TIMED_REQUESTS)]
    for row in rows[:WARMUP_REQUESTS]:
        model.predict(row)
    latencies = []
    for row in rows[WARMUP_REQUESTS:]:
        t = time.perf_counter()
        model.predict(row)
        latencies.append(time.perf_counter() - t)
    lat_ms = np.array(latencies) * 1000

    batch = X.iloc[:BATCH_SIZE]
    model.predict(batch)  # warm-up
    batch_times = []
    for _ in range(BATCH_REPEATS):
        t = time.perf_counter()
        model.predict(batch)
        batch_times.append(time.perf_counter() - t)
    batch_s = float(np.median(batch_times))

    return {
        "load_s": load_s,
        "rss_before_load_mb": rss_before_mb,
        "rss_after_load_mb": rss_after_load_mb,
        "peak_ram_mb": _peak_rss_mb(),
        "latency_p50_ms": float(np.percentile(lat_ms, 50)),
        "latency_p95_ms": float(np.percentile(lat_ms, 95)),
        "latency_p99_ms": float(np.percentile(lat_ms, 99)),
        "single_throughput_rps": float(TIMED_REQUESTS / (lat_ms.sum() / 1000)),
        "batch_1000_ms": batch_s * 1000,
        "batch_throughput_rows_s": BATCH_SIZE / batch_s,
    }


def measure_in_fresh_process(model_path: Path, bench_path: Path, timeout_s: int = 1800) -> dict:
    """Run run_benchmark() in a new Python process, so the RAM numbers only contain this model."""
    cmd = [sys.executable, "-m", "src.serving_selection", "measure",
           "--model", str(model_path), "--bench", str(bench_path)]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed for {model_path}:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def benchmark_candidates(results: pd.DataFrame, bench_path: Path) -> pd.DataFrame:
    """Benchmark every candidate (smallest first) in its own process; adds the measured columns."""
    measured = []
    for row in results.sort_values("size_mb").itertuples():
        print(f"benchmarking {row.candidate} ({row.size_mb:,.0f} MB) ...", flush=True)
        try:
            m = measure_in_fresh_process(Path(row.model_path), bench_path)
        except Exception as exc:  # e.g. out of memory for the largest forests
            print(f"  failed: {exc}")
            m = {"benchmark_error": str(exc)[-300:]}
        measured.append({"candidate": row.candidate, **m})
    return results.merge(pd.DataFrame(measured), on="candidate", how="left")


# ---------------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionPolicy:
    """Written down before any candidate is measured (see the notebook).

    1. Accuracy: validation RMSE within rmse_tolerance (1%) of the best candidate.
    2. Size: serialized skops model below max_size_mb (700 MB).
    3. Among the remaining candidates: lowest peak RAM, then lowest p95 latency.
    """

    rmse_tolerance: float = 0.01
    max_size_mb: float = 700.0


def apply_policy(results: pd.DataFrame, policy: SelectionPolicy = SelectionPolicy()) -> pd.DataFrame:
    """Adds 'status' (selected / eligible / rejected) and 'reason' to every candidate.

    Candidates without benchmark numbers cannot be ranked on RAM/latency and are rejected.
    """
    df = results.copy()
    best_rmse = df["val_rmse"].min()
    rmse_limit = best_rmse * (1 + policy.rmse_tolerance)
    df["rmse_vs_best_pct"] = (df["val_rmse"] / best_rmse - 1) * 100

    reasons = []
    for row in df.itertuples():
        why = []
        if row.val_rmse > rmse_limit:
            why.append(f"RMSE {row.val_rmse:.4f} is {row.rmse_vs_best_pct:.2f}% above best "
                       f"{best_rmse:.4f} (limit {policy.rmse_tolerance:.0%})")
        if row.size_mb >= policy.max_size_mb:
            why.append(f"size {row.size_mb:,.0f} MB >= {policy.max_size_mb:,.0f} MB budget")
        if pd.isna(getattr(row, "peak_ram_mb", np.nan)):
            why.append("no benchmark result")
        reasons.append("; ".join(why))
    df["reason"] = reasons
    df["status"] = np.where(df["reason"] == "", "eligible", "rejected")

    if not (df["status"] == "eligible").any():
        raise ValueError("No candidate passes the selection rule - revisit the candidates.")
    eligible = df[df["status"] == "eligible"].sort_values(["peak_ram_mb", "latency_p95_ms"])
    winner = eligible.index[0]
    df.loc[winner, "status"] = "selected"
    df.loc[winner, "reason"] = "passes accuracy and size limits; lowest peak RAM"
    w = df.loc[winner]
    for idx in eligible.index[1:]:
        r = df.loc[idx]
        if r["peak_ram_mb"] > w["peak_ram_mb"]:
            df.loc[idx, "reason"] = (f"eligible, but peak RAM {r['peak_ram_mb']:,.0f} MB > "
                                     f"{w['peak_ram_mb']:,.0f} MB of {w['candidate']}")
        else:
            df.loc[idx, "reason"] = (f"eligible, same peak RAM but p95 {r['latency_p95_ms']:.2f} ms >= "
                                     f"{w['latency_p95_ms']:.2f} ms of {w['candidate']}")
    return df.sort_values(["status", "val_rmse"], key=lambda s: s.map(
        {"selected": 0, "eligible": 1, "rejected": 2}) if s.name == "status" else s)


# ---------------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------------


def log_datasets(datasets: dict[str, pd.DataFrame], source: str) -> None:
    """Log dataset information into the ACTIVE MLflow run.

    datasets: {"training": train_df, "validation": val_df, ...}; each frame must contain
    tpep_pickup_datetime. Logs the period and size as params (<context>_start, _end, _rows)
    and the schema/profile as an MLflow dataset input (no data is uploaded).
    """
    import mlflow
    import mlflow.data

    for context, df in datasets.items():
        pickup = df["tpep_pickup_datetime"]
        mlflow.log_params({
            f"{context}_start": f"{pickup.min():%Y-%m-%d}",
            f"{context}_end": f"{pickup.max():%Y-%m-%d}",
            f"{context}_rows": len(df),
        })
        ds = mlflow.data.from_pandas(df, source=source, name=f"yellow-taxi-{context}")
        mlflow.log_input(ds, context=context)


def latest_run(run_name: str, experiment: str = "nyc-taxi-duration"):
    """The most recent FINISHED run with this name (an mlflow Run object)."""
    import mlflow

    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise LookupError(f"No finished '{run_name}' run in MLflow experiment '{experiment}'. "
                          "Run `uv run nyc-taxi all` first (MLflow server must be running).")
    return runs[0]


def child_runs(parent_run_id: str, experiment: str = "nyc-taxi-duration") -> pd.DataFrame:
    """All child runs of a parent run as a DataFrame (columns params.*, metrics.*, tags.*)."""
    import mlflow

    return mlflow.search_runs(experiment_names=[experiment],
                              filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'")


def download_run_file(run_id: str, name: str) -> Path:
    """Download one artifact file of a run (not the whole run) and return its local path."""
    import mlflow

    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name))


def registered_version(run_id: str, model_name: str) -> str | None:
    """Model-registry version created from this run, if any."""
    from mlflow import MlflowClient

    versions = [v for v in MlflowClient().search_model_versions(f"run_id='{run_id}'")
                if v.name == model_name]
    return str(max(int(v.version) for v in versions)) if versions else None


def alias_version(model_name: str, alias: str) -> str | None:
    """Version the alias points to, or None if the alias (or model) does not exist."""
    from mlflow import MlflowClient

    try:
        return str(MlflowClient().get_model_version_by_alias(model_name, alias).version)
    except Exception:
        return None


# ---------------------------------------------------------------------------------
# Full selection run (training happens here, never in the notebooks)
# ---------------------------------------------------------------------------------


def selection_rule_text(policy: SelectionPolicy) -> str:
    """The rule, data rules and measurement conditions, as logged to MLflow (selection_rule.txt)."""
    return f"""Selection rule (fixed before any candidate is measured)
1. Accuracy: validation RMSE within {policy.rmse_tolerance:.0%} of the best candidate's validation RMSE.
2. Size: serialized model (skops, the format logged to MLflow) under {policy.max_size_mb:,.0f} MB.
3. Among the candidates that pass 1 and 2: lowest peak RAM, then lowest p95 latency.

Data rules
- Candidates are trained on Jan 1-20 and compared on validation, Jan 21-25.
- The January test days (Jan 26-31) were used by src.train (rf-final-jan1-25), so they are not used here.
- Only the selected configuration is retrained on all of January and evaluated once on February 2025.
- Statistics learned from trips (baseline medians, zone-pair counts) come from the training period only.

Measurement conditions (same for every candidate, one candidate per fresh Python process)
- model loaded from its .skops file, n_jobs=1, {WARMUP_REQUESTS} warm-up requests
- {TIMED_REQUESTS:,} sequential single-row predictions (1-row DataFrame) -> p50/p95/p99 latency
- batch of {BATCH_SIZE:,} rows, median of {BATCH_REPEATS} runs -> batch time and rows/s
- peak RAM = peak resident memory of that process (Python + libraries + model + predictions)
- model time only: feature building and HTTP are not included
"""


BASELINE_LINE_STYLES = {"Overall median": ":", "Zone-pair median": "-.", "RF baseline": "--"}
OFF_CHART_FACTOR = 1.5  # baselines above 1.5x the worst candidate RMSE are listed, not drawn


def plot_tradeoff(ranked: pd.DataFrame, xcol: str, xlabel: str, title: str,
                  policy: SelectionPolicy, size_budget: bool = False,
                  baselines: dict[str, float] | None = None, path: Path | None = None):
    """Scatter of validation RMSE against one serving cost, coloured by status.

    baselines: {label: validation RMSE} drawn as horizontal reference lines (the baseline
    runs from src.train, scored on the same validation days). Uses matplotlib's Figure
    directly (no pyplot), so it works in the CLI and in a notebook (display(fig)).
    Saves a PNG if path is given. Returns the figure.
    """
    from matplotlib.figure import Figure

    colors = ranked["status"].map({"selected": "tab:green", "eligible": "tab:blue",
                                   "rejected": "tab:grey"})
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.subplots()
    ax.scatter(ranked[xcol], ranked["val_rmse"], c=colors, s=60)
    for row in ranked.itertuples():
        ax.annotate(row.candidate, (getattr(row, xcol), row.val_rmse), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.axhline(ranked["val_rmse"].min() * (1 + policy.rmse_tolerance), color="tab:red",
               linestyle="--", label=f"best RMSE + {policy.rmse_tolerance:.0%}")
    worst = ranked["val_rmse"].max()
    for label, rmse in (baselines or {}).items():
        if not np.isfinite(rmse):
            continue
        style = {"color": "black", "linewidth": 1, "linestyle": BASELINE_LINE_STYLES.get(label, ":")}
        if rmse <= OFF_CHART_FACTOR * worst:
            ax.axhline(rmse, label=f"{label} ({rmse:.3f})", **style)
        else:  # far above every candidate: a line would squash the candidates, list it instead
            ax.plot([], [], label=f"{label} ({rmse:.3f}, above chart)", **style)
    if size_budget:
        ax.axvline(policy.max_size_mb, color="tab:orange", linestyle="--",
                   label=f"size budget {policy.max_size_mb:,.0f} MB")
    ax.set(xlabel=xlabel, ylabel="validation RMSE (min)", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def baseline_val_rmse(experiment: str = "nyc-taxi-duration") -> dict[str, float]:
    """{label: val_rmse} of the baseline runs logged by src.train; prints a note if any is missing."""
    from src.baselines import baseline_metrics_table, missing_runs_message, read_baseline_runs

    runs, missing = read_baseline_runs(experiment)
    if missing:
        print(missing_runs_message(missing))
    if not runs:
        return {}
    table = baseline_metrics_table(runs)
    return table["val_rmse"].dropna().to_dict() if "val_rmse" in table else {}


def _finite_metrics(values: dict) -> dict[str, float]:
    """Keep numeric, finite values (MLflow metrics)."""
    out = {}
    for k, v in values.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            if np.isfinite(v):
                out[k] = float(v)
    return out


def log_candidate_runs(ranked: pd.DataFrame) -> None:
    """One nested child run per candidate: params, all numeric columns as metrics, status tags."""
    import mlflow

    param_cols = ["family", "n_trees", "max_depth", "min_samples_leaf", "max_features"]
    for row in ranked.to_dict("records"):
        with mlflow.start_run(run_name=row["candidate"], nested=True):
            mlflow.log_params({k: row[k] for k in param_cols if k in row})
            mlflow.log_metrics(_finite_metrics({k: v for k, v in row.items() if k not in param_cols}))
            mlflow.set_tags({"status": row["status"], "reason": row["reason"][:5000]})
            if isinstance(row.get("benchmark_error"), str):
                mlflow.set_tag("benchmark_error", row["benchmark_error"])


def holdout_metrics(model, train_df: pd.DataFrame, holdout_df: pd.DataFrame,
                    zone_xy: pd.DataFrame) -> dict[str, float]:
    """February metrics of the serving model and of the zone-pair median (same training trips).

    Keys: holdout_rmse/mae, holdout_<segment>_rmse/mae, holdout_<segment>_trips,
    holdout_zone_pair_median_rmse/mae and holdout_zone_pair_median_<segment>_rmse/mae.
    """
    from src.features import build_features

    pred = model.predict(build_features(holdout_df, zone_xy))
    base = zone_pair_median_predict(train_df, holdout_df)
    masks = segment_masks(holdout_df, zone_pair_counts(train_df))
    out = {f"holdout_{k}": v for k, v in regression_metrics(holdout_df["duration"], pred).items()}
    out.update({f"holdout_{k}": v for k, v in segment_metrics(holdout_df, pred, masks).items()})
    out.update({f"holdout_{k}_trips": float(n) for k, n in segment_sizes(masks).items()})
    out.update(regression_metrics(holdout_df["duration"], base, prefix="holdout_zone_pair_median_"))
    out.update({f"holdout_zone_pair_median_{k}": v
                for k, v in segment_metrics(holdout_df, base, masks).items()})
    return out


def machine_info() -> dict[str, str]:
    """Hardware and library versions of the benchmark machine (logged as params)."""
    import platform

    import psutil
    import sklearn

    return {
        "machine_cpu": f"{platform.processor() or platform.machine()} "
                       f"({psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads)",
        "machine_ram_gb": f"{psutil.virtual_memory().total / 2**30:.1f}",
        "machine_os": platform.platform(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }


def run_selection(policy: SelectionPolicy = SelectionPolicy(), out_dir: Path = OUTPUT_DIR) -> dict:
    """Train, benchmark and select the serving model; evaluate it on February; register it.

    Everything is logged to the MLflow parent run "serving-selection" (see the module
    docstring). The champion alias is never moved: the run fails if it changed.
    Returns {"run_id", "selected", "registered_version"}.
    """
    import gc

    import mlflow
    import skops.io as sio

    from src.experiment import EXPERIMENT, TRACKING_URI, log_model_artifacts
    from src.features import build_features, build_zone_xy
    from src.train import DATA_URL, MODEL_NAME, load_january, load_tuned_params, split_by_time

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)  # fails fast if the MLflow server is not running
    tuned_params, params_source = load_tuned_params()  # latest optuna-tuning run, else BEST_PARAMS
    out_dir.mkdir(parents=True, exist_ok=True)
    champion_before = alias_version(MODEL_NAME, "champion")

    # Data: train Jan 1-20, validation Jan 21-25 (the January test days are dropped)
    jan = load_january()
    tr, va, _ = split_by_time(jan)
    train_df, val_df = jan[tr], jan[va]
    del jan
    zone_xy = build_zone_xy(DATA_DIR)  # fixed geography, not learned from trips
    X_train, y_train = build_features(train_df, zone_xy), train_df["duration"]
    X_val = build_features(val_df, zone_xy)
    masks = segment_masks(val_df, zone_pair_counts(train_df))
    baselines = baseline_val_rmse(EXPERIMENT)  # reference lines on the charts (runs from src.train)
    print("train:     ", describe_period(train_df))
    print("validation:", describe_period(val_df))

    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent:
        mlflow.log_params({"rmse_tolerance": policy.rmse_tolerance, "max_size_mb": policy.max_size_mb,
                           **{f"reference_{k}": v for k, v in tuned_params.items()},
                           "params_source": params_source, **machine_info()})
        log_datasets({"training": train_df, "validation": val_df}, source=DATA_URL)

        # 1. Candidates (one fit per family; a candidate is the first n trees of its family)
        families = candidate_families(tuned_params)
        parts = []
        for family in families:
            print(f"training {family.name}: {family.params}, up to {max(family.tree_counts)} trees ...",
                  flush=True)
            parts.append(train_family(family, X_train, y_train, X_val, val_df, masks,
                                      model_dir=out_dir / "models"))
            gc.collect()
        results = pd.concat(parts, ignore_index=True)

        # 2. Fresh-process benchmarks
        bench_path = write_bench_features(X_val, out_dir / "bench_features.parquet")
        results = benchmark_candidates(results, bench_path)

        # 3. Selection rule, child runs and comparison artifacts
        ranked = apply_policy(results, policy)
        selected = ranked[ranked["status"] == "selected"].iloc[0]
        print(f"selected: {selected['candidate']} (val RMSE {selected['val_rmse']:.4f}, "
              f"{selected['size_mb']:,.0f} MB, peak RAM {selected['peak_ram_mb']:,.0f} MB)")
        log_candidate_runs(ranked)

        table_path = out_dir / TABLE_FILE
        ranked.to_csv(table_path, index=False)
        rule_path = out_dir / RULE_FILE
        rule_path.write_text(selection_rule_text(policy), encoding="utf-8")
        charts = [out_dir / SIZE_CHART_FILE, out_dir / LATENCY_CHART_FILE]
        plot_tradeoff(ranked, "size_mb", "serialized size (MB, skops)", "Accuracy vs size", policy,
                      size_budget=True, baselines=baselines, path=charts[0])
        plot_tradeoff(ranked, "latency_p95_ms", "single-request p95 latency (ms)", "Accuracy vs latency",
                      policy, baselines=baselines, path=charts[1])
        for path in (table_path, rule_path, *charts):
            mlflow.log_artifact(str(path))
        mlflow.set_tag("selected_candidate", selected["candidate"])
        mlflow.log_metrics(_finite_metrics({
            "selected_val_rmse": selected["val_rmse"], "selected_val_mae": selected["val_mae"],
            **{f"selected_{k}": selected[k] for k in BENCHMARK_COLUMNS}}))

        # 4. Repeatability: refit from scratch with exactly n trees, compare with the measured model
        family = next(f for f in families if f.name == selected["family"])
        n_trees = int(selected["n_trees"])
        selected_params = {**family.params, "n_estimators": n_trees}
        mlflow.log_params({f"selected_{k}": v for k, v in selected_params.items()})
        measured_model = sio.load(selected["model_path"], trusted=SKOPS_TRUSTED_TYPES)
        repeat = check_repeatability(family.params, n_trees, X_train, y_train, X_val,
                                     measured_model.predict(X_val))
        del measured_model
        print("repeatability:", repeat)
        mlflow.log_metric("repeat_max_abs_diff", repeat["max_abs_diff_fresh_vs_sliced"])
        mlflow.set_tag("repeatable", str(repeat["identical"]))
        if not repeat["identical"]:
            raise RuntimeError("Training is not repeatable - investigate before deploying")
        del X_train, y_train, X_val, train_df, val_df
        gc.collect()

        # 5. Train on all of January, evaluate once on February 2025
        jan_all = load_january()
        feb = load_month(2025, 2)
        print("final training:", describe_period(jan_all))
        print("holdout:       ", describe_period(feb))
        X_all = build_features(jan_all, zone_xy)
        serving_model = fit_rf(family.params, X_all, jan_all["duration"], n_estimators=n_trees)
        holdout = holdout_metrics(serving_model, jan_all, feb, zone_xy)
        holdout["serving_model_size_mb"] = save_skops(serving_model, out_dir / "serving_model.skops")
        mlflow.log_metrics(_finite_metrics(holdout))
        log_datasets({"final_training": jan_all, "holdout": feb}, source=DATA_URL)
        print(f"February holdout: RMSE {holdout['holdout_rmse']:.4f}, MAE {holdout['holdout_mae']:.4f} "
              f"(zone-pair median RMSE {holdout['holdout_zone_pair_median_rmse']:.4f})")

        # 6. Register WITHOUT an alias (model + zone table in this parent run, as the API expects)
        from src.bundle import period_of

        version = log_model_artifacts(serving_model, zone_xy=zone_xy, register_name=MODEL_NAME,
                                      X=X_all, training_period=period_of(jan_all))
        mlflow.set_tag("registered_version", version)
        champion_after = alias_version(MODEL_NAME, "champion")
        if champion_after != champion_before:
            raise RuntimeError("champion alias changed - it must only be moved by hand")

    print(f"Registered {MODEL_NAME} version {version} (run {parent.info.run_id}); "
          f"champion still -> {champion_after}")
    return {"run_id": parent.info.run_id, "selected": selected["candidate"],
            "registered_version": version}


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="train, benchmark, select, evaluate on February and register (no alias)")
    m = sub.add_parser("measure", help="benchmark one skops model in this (fresh) process")
    m.add_argument("--model", type=Path, required=True)
    m.add_argument("--bench", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_selection()
    elif args.command == "measure":
        print(json.dumps(run_benchmark(args.model, args.bench)))


if __name__ == "__main__":
    main()
