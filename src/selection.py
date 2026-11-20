"""The `nyc-taxi select` step: one simple rule, one test score, one registered model.

    uv run nyc-taxi select     (or uv run python -m src.selection)

Rule (fixed before the results; also written in docs/operations.md and the notebook):
    Candidates: the RF baseline ("rf-baseline") and every Optuna trial ("optuna-tuning"),
    all fitted on the training sample (Jan 1-20) and scored on validation (Jan 21-25).
    1. best   = the lowest validation RMSE
    2. tied   = every candidate within 1% of the best validation RMSE
    3. winner = the smallest model (model_mb) among the tied candidates

The winner's settings are fitted again on the same training sample with the same seed
(so it is the same model), scored ONCE on test (Jan 26-31), logged with its zone table
and feature schema and registered WITHOUT an alias. The baselines are scored on the same
test days, so the final table compares everything on the same trips.

Logged as one MLflow run "select": candidates.csv (every candidate, tied, winner),
test_comparison.csv, selection_rule.txt, rmse_vs_size.png, the winner's metrics, and
the registered model. The champion alias is never moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import evaluation as ev
from src.baselines import NAIVE_BASELINES, OVERALL_MEDIAN_RUN, RF_BASELINE_RUN, ZONE_PAIR_MEDIAN_RUN

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "selection"  # git-ignored
SELECT_RUN_NAME = "select"
RMSE_TOLERANCE = 0.01
PARAM_KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "max_features")

CANDIDATES_FILE = "candidates.csv"
TEST_FILE = "test_comparison.csv"
RULE_FILE = "selection_rule.txt"
CHART_FILE = "rmse_vs_size.png"

NAIVE_LABELS = {OVERALL_MEDIAN_RUN: "Overall median", ZONE_PAIR_MEDIAN_RUN: "Zone-pair median"}
OFF_CHART_FACTOR = 1.5  # naive baselines above 1.5x the worst candidate RMSE are listed, not drawn


def rule_text(tolerance: float = RMSE_TOLERANCE) -> str:
    return f"""Selection rule (fixed before the results)
Candidates: the RF baseline and every Optuna trial, all fitted on the training sample (Jan 1-20)
and scored on validation (Jan 21-25). The naive baselines are references, not candidates.
1. Best: the lowest validation RMSE.
2. Tied: every candidate whose validation RMSE is within {tolerance:.0%} of the best.
3. Winner: the smallest model (model_mb, skops size) among the tied candidates.
Then the winner's settings are fitted again on the same training sample with the same seed
(the same model), scored once on test (Jan 26-31) and registered without an alias.
The baselines are scored on the same test days for comparison.
"""


# ---------------------------------------------------------------------------------
# Candidates and the rule (pure functions, unit-tested)
# ---------------------------------------------------------------------------------


def _candidate_row(name: str, run_id: str, metrics: dict, params: dict) -> dict[str, Any]:
    return {
        "candidate": name,
        "run_id": run_id,
        "val_rmse": metrics.get("val_rmse", np.nan),
        "val_mae": metrics.get("val_mae", np.nan),
        "model_mb": metrics.get("model_mb", np.nan),
        **{key: params.get(key) for key in PARAM_KEYS},
    }


def collect_candidates(experiment: str = "nyc-taxi-duration") -> pd.DataFrame:
    """The latest rf-baseline run and every trial of the latest optuna-tuning run.

    Runs without val_rmse or model_mb (older runs) are left out.
    """
    from src.tracking import child_runs, latest_run
    from src.tuning import TUNING_RUN_NAME

    rf = latest_run(RF_BASELINE_RUN, experiment)
    rows = [_candidate_row(RF_BASELINE_RUN, rf.info.run_id, dict(rf.data.metrics), dict(rf.data.params))]
    tuning = latest_run(TUNING_RUN_NAME, experiment)
    for record in child_runs(tuning.info.run_id, experiment).to_dict("records"):
        metrics = {k.removeprefix("metrics."): v for k, v in record.items() if k.startswith("metrics.")}
        params = {k.removeprefix("params."): v for k, v in record.items() if k.startswith("params.")}
        rows.append(_candidate_row(record["tags.mlflow.runName"], record["run_id"], metrics, params))

    candidates = pd.DataFrame(rows).dropna(subset=["val_rmse", "model_mb"]).reset_index(drop=True)
    if candidates.empty:
        raise LookupError(
            "No candidate run has val_rmse and model_mb. "
            "Run `uv run nyc-taxi train` and `uv run nyc-taxi tune`."
        )
    return candidates


def apply_rule(candidates: pd.DataFrame, tolerance: float = RMSE_TOLERANCE) -> pd.DataFrame:
    """Adds rmse_vs_best_pct, tied and winner; winner first, then tied, then the rest by RMSE."""
    df = candidates.copy()
    best = df["val_rmse"].min()
    df["rmse_vs_best_pct"] = (df["val_rmse"] / best - 1) * 100
    df["tied"] = df["val_rmse"] <= best * (1 + tolerance)
    winner = df[df["tied"]].sort_values(["model_mb", "val_rmse"]).index[0]
    df["winner"] = df.index == winner
    return df.sort_values(["winner", "tied", "val_rmse"], ascending=[False, False, True]).reset_index(
        drop=True
    )


def candidate_params(row: pd.Series | dict) -> dict[str, int | float]:
    """Random Forest params of a candidate (MLflow stores them as text).

    max_features stays a float (1.0 = all features; the integer 1 would mean one feature).
    Params a run did not set are left out, so scikit-learn's defaults apply as in that run.
    """
    params: dict[str, int | float] = {}
    for key in PARAM_KEYS:
        value = row.get(key)
        if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
            continue
        params[key] = float(value) if key == "max_features" else int(float(value))
    return params


# ---------------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------------


def plot_rmse_vs_size(
    table: pd.DataFrame,
    tolerance: float = RMSE_TOLERANCE,
    baselines: dict[str, float] | None = None,
    path: Path | None = None,
):
    """Validation RMSE against model size: winner green, tied blue, others grey.

    baselines: {label: validation RMSE} of the naive baselines, drawn as reference lines
    (listed in the legend instead when far above every candidate). Uses matplotlib's
    Figure directly (no pyplot), so it works in the CLI and in a notebook.
    """
    from matplotlib.figure import Figure

    colors = np.where(table["winner"], "tab:green", np.where(table["tied"], "tab:blue", "tab:grey"))
    fig = Figure(figsize=(8, 5.5))
    ax = fig.subplots()
    ax.scatter(table["model_mb"], table["val_rmse"], c=colors, s=60)
    for row in table.itertuples():
        ax.annotate(
            row.candidate, (row.model_mb, row.val_rmse), fontsize=7, xytext=(4, 3), textcoords="offset points"
        )
    best = table["val_rmse"].min()
    ax.axhline(
        best * (1 + tolerance), color="tab:red", linestyle="--", label=f"best + {tolerance:.0%} (tied below)"
    )
    worst = table["val_rmse"].max()
    for label, rmse in (baselines or {}).items():
        if not np.isfinite(rmse):
            continue
        style: dict[str, Any] = {"color": "black", "linewidth": 1, "linestyle": ":"}
        if rmse <= OFF_CHART_FACTOR * worst:
            ax.axhline(rmse, label=f"{label} ({rmse:.3f})", **style)
        else:  # far above every candidate: a line would squash the candidates, list it instead
            ax.plot([], [], label=f"{label} ({rmse:.3f}, above chart)", **style)
    ax.set_xscale("log")
    ax.set(xlabel="model size (MB, skops, log scale)", ylabel="validation RMSE (min)", title="RMSE vs size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


# ---------------------------------------------------------------------------------
# The select step
# ---------------------------------------------------------------------------------


def _slug(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def run_selection(tolerance: float = RMSE_TOLERANCE, out_dir: Path = OUTPUT_DIR) -> dict:
    """Apply the rule, refit the winner, score it once on test, register it (no alias)."""
    import mlflow

    from src.baselines import read_baseline_runs
    from src.bundle import period_of
    from src.experiment import EXPERIMENT, TRACKING_URI, log_model_artifacts
    from src.features import build_features, build_zone_xy
    from src.tracking import alias_version
    from src.train import MODEL_NAME, load_splits, period_params

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)  # fails fast if the MLflow server is not running
    champion_before = alias_version(MODEL_NAME, "champion")

    # 1. The rule, over the runs of `train` and `tune` (validation only)
    table = apply_rule(collect_candidates(EXPERIMENT), tolerance)
    winner = table.iloc[0]
    params = candidate_params(winner)
    print(
        f"{int(table['tied'].sum())} of {len(table)} candidates tied within {tolerance:.0%} of the best RMSE"
    )
    print(
        f"winner: {winner['candidate']} {params} "
        f"(val RMSE {winner['val_rmse']:.4f}, {winner['model_mb']:.1f} MB)"
    )

    # 2. Refit the winner on the same training sample (same seed = same model); test once
    splits = load_splits()
    zone_xy = build_zone_xy(ROOT / "data")
    X_sample, y_sample = build_features(splits.sample, zone_xy), splits.sample["duration"]
    X_test = build_features(splits.test, zone_xy)
    model = ev.fit_rf(params, X_sample, y_sample)
    model_mb = ev.model_size_mb(model)
    test_metrics = ev.score_predictions("test", splits.train, splits.test, model.predict(X_test))

    # 3. Baselines on the same test days (naive: training-day statistics; RF baseline refitted)
    baseline_runs, _ = read_baseline_runs(EXPERIMENT)
    rows = []
    for run_name, (predict, _description) in NAIVE_BASELINES.items():
        label = NAIVE_LABELS[run_name]
        scores = ev.score_predictions("test", splits.train, splits.test, predict(splits.train, splits.test))
        val = baseline_runs[label].data.metrics.get("val_rmse", np.nan) if label in baseline_runs else np.nan
        rows.append({"model": label, "val_rmse": val, **scores})
    rf_row = table[table["candidate"] == RF_BASELINE_RUN].iloc[0]
    rf_model = (
        model
        if winner["candidate"] == RF_BASELINE_RUN
        else ev.fit_rf(candidate_params(rf_row), X_sample, y_sample)
    )
    rf_scores = ev.score_predictions("test", splits.train, splits.test, rf_model.predict(X_test))
    rows.append({"model": "RF baseline", "val_rmse": rf_row["val_rmse"], **rf_scores})
    del rf_model
    winner_label = f"Winner ({winner['candidate']})"
    rows.append({"model": winner_label, "val_rmse": winner["val_rmse"], **test_metrics})
    comparison = pd.DataFrame(rows).set_index("model")

    # 4. Files for the notebook
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name in (CANDIDATES_FILE, TEST_FILE, RULE_FILE, CHART_FILE)}
    table.to_csv(paths[CANDIDATES_FILE], index=False)
    comparison.to_csv(paths[TEST_FILE])
    paths[RULE_FILE].write_text(rule_text(tolerance), encoding="utf-8")
    naive_val = {
        label: float(r["val_rmse"]) for label, r in comparison.iterrows() if label in NAIVE_LABELS.values()
    }
    plot_rmse_vs_size(table, tolerance, naive_val, path=paths[CHART_FILE])

    # 5. Log and register WITHOUT an alias
    with mlflow.start_run(run_name=SELECT_RUN_NAME) as run:
        mlflow.log_params(
            {
                "rule_tolerance": tolerance,
                "winner": winner["candidate"],
                "winner_run_id": winner["run_id"],
                "candidates": len(table),
                "tied": int(table["tied"].sum()),
                **{f"winner_{k}": v for k, v in params.items()},
                **period_params(splits),
            }
        )
        metrics = {
            "winner_val_rmse": float(winner["val_rmse"]),
            "winner_val_mae": float(winner["val_mae"]),
            "winner_model_mb": model_mb,
            **test_metrics,
        }
        for label, r in comparison.drop(index=winner_label).iterrows():
            metrics[f"test_{_slug(str(label))}_rmse"] = float(r["test_rmse"])
            metrics[f"test_{_slug(str(label))}_mae"] = float(r["test_mae"])
        mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
        mlflow.set_tag("winner", winner["candidate"])
        for path in paths.values():
            mlflow.log_artifact(str(path))
        version = log_model_artifacts(
            model,
            zone_xy=zone_xy,
            register_name=MODEL_NAME,
            X=X_sample,
            training_period=period_of(splits.sample),
        )
        mlflow.set_tag("registered_version", str(version))
        champion_after = alias_version(MODEL_NAME, "champion")
        if champion_after != champion_before:
            raise RuntimeError("champion alias changed - it must only be moved by hand")

    print(
        f"Winner: {winner['candidate']} | val RMSE {winner['val_rmse']:.4f} | "
        f"test RMSE {test_metrics['test_rmse']:.4f} | {model_mb:.1f} MB | "
        f"registered {MODEL_NAME} version {version} (no alias; champion still -> {champion_after})"
    )
    return {
        "run_id": run.info.run_id,
        "selected": winner["candidate"],
        "registered_version": version,
        "val_rmse": float(winner["val_rmse"]),
        "test_rmse": test_metrics["test_rmse"],
        "model_mb": model_mb,
    }


if __name__ == "__main__":
    run_selection()
