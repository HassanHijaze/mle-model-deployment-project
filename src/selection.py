"""Model selection: validation-only ranking, one final test, MLflow registration.

Run:
    uv run nyc-taxi select

Selection rule:
    1. Find the lowest validation RMSE.
    2. Keep candidates within 1% of that RMSE.
    3. Select the smallest serialized model among those candidates.

The selected model is refitted on the same training sample and evaluated
on the test set only AFTER selection.

Naive baselines are reference models, not serving candidates.
The MLflow champion alias is never changed automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import evaluation as ev
from src.baselines import (
    NAIVE_BASELINES,
    OVERALL_MEDIAN_RUN,
    RF_BASELINE_RUN,
    ZONE_PAIR_MEDIAN_RUN,
)
from src.experiment import EXPERIMENT

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "selection"

SELECT_RUN_NAME = "select"
RMSE_TOLERANCE = 0.01

PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
)

CANDIDATES_FILE = "candidates.csv"
TEST_FILE = "test_comparison.csv"
RULE_FILE = "selection_rule.txt"
CHART_FILE = "rmse_vs_size.png"

NAIVE_LABELS = {
    OVERALL_MEDIAN_RUN: "Overall median",
    ZONE_PAIR_MEDIAN_RUN: "Zone-pair median",
}


# ------------------------------------------------------------
# 1. Selection policy
# ------------------------------------------------------------


def rule_text(tolerance: float = RMSE_TOLERANCE) -> str:
    return f"""Model selection policy

Training: January 1-20, 2025
Validation: January 21-25, 2025
Testing: January 26-31, 2025

Candidates:
- Random Forest baseline
- Every Optuna trial

Selection:
1. Find the lowest validation RMSE.
2. Keep candidates within {tolerance:.0%} of the best RMSE.
3. Select the smallest serialized model among them.
4. Break size ties using validation RMSE.

The selected model is refitted on the same training
sample using the same random seed.

Only after selection, evaluate on the test set.

Naive baselines are references, not serving candidates.

Register the selected model without changing champion.
"""


# ------------------------------------------------------------
# 2. Collect candidates
# ------------------------------------------------------------


def _candidate_row(
    name: str,
    run_id: str,
    metrics: dict,
    params: dict,
) -> dict[str, Any]:

    return {
        "candidate": name,
        "run_id": run_id,
        "val_rmse": metrics.get("val_rmse", np.nan),
        "val_mae": metrics.get("val_mae", np.nan),
        "model_mb": metrics.get("model_mb", np.nan),
        **{key: params.get(key) for key in PARAM_KEYS},
    }


def collect_candidates(
    experiment: str = EXPERIMENT,
) -> pd.DataFrame:
    """Collect RF baseline and Optuna candidates from MLflow."""

    from src.tracking import child_runs, latest_run
    from src.tuning import TUNING_RUN_NAME

    rf = latest_run(RF_BASELINE_RUN, experiment)

    rows = [
        _candidate_row(
            RF_BASELINE_RUN,
            rf.info.run_id,
            dict(rf.data.metrics),
            dict(rf.data.params),
        )
    ]

    tuning = latest_run(TUNING_RUN_NAME, experiment)

    trials = child_runs(
        tuning.info.run_id,
        experiment,
    )

    for record in trials.to_dict("records"):
        metrics = {k.removeprefix("metrics."): v for k, v in record.items() if k.startswith("metrics.")}

        params = {k.removeprefix("params."): v for k, v in record.items() if k.startswith("params.")}

        rows.append(
            _candidate_row(
                record["tags.mlflow.runName"],
                record["run_id"],
                metrics,
                params,
            )
        )

    df = pd.DataFrame(rows)

    df = df.dropna(subset=["val_rmse", "model_mb"]).reset_index(drop=True)

    if df.empty:
        raise LookupError("No valid candidates found. Run nyc-taxi train and nyc-taxi tune first.")

    return df


# ------------------------------------------------------------
# 3. Apply selection rule
# ------------------------------------------------------------


def apply_rule(
    candidates: pd.DataFrame,
    tolerance: float = RMSE_TOLERANCE,
) -> pd.DataFrame:
    """Select the smallest model within the RMSE tolerance."""

    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be a fraction between 0 and 1.")

    df = candidates.copy()

    if df.empty:
        raise ValueError("Candidate table is empty.")

    for column in ("val_rmse", "model_mb"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(df[["val_rmse", "model_mb"]].to_numpy()).all():
        raise ValueError("Candidate metrics must be finite.")

    if (df["val_rmse"] <= 0).any():
        raise ValueError("RMSE must be positive.")

    if (df["model_mb"] <= 0).any():
        raise ValueError("Model sizes must be positive.")

    best = df["val_rmse"].min()

    df["rmse_vs_best_pct"] = (df["val_rmse"] / best - 1) * 100

    df["tied"] = df["val_rmse"] <= best * (1 + tolerance)

    tied = df.loc[df["tied"]]

    winner_index = tied.sort_values(["model_mb", "val_rmse"]).index[0]

    df["winner"] = df.index == winner_index

    return df.sort_values(
        ["winner", "tied", "val_rmse"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def candidate_params(
    row: pd.Series | dict,
) -> dict[str, int | float]:
    """Recover Random Forest hyperparameters from MLflow."""

    params = {}

    for key in PARAM_KEYS:
        value = row.get(key)

        if value is None or value == "":
            continue

        if pd.isna(value):
            continue

        if key == "max_features":
            params[key] = float(value)
        else:
            params[key] = int(float(value))

    return params


# ------------------------------------------------------------
# 4. Professional README visualization
# ------------------------------------------------------------


def plot_rmse_vs_size(
    table: pd.DataFrame,
    tolerance: float = RMSE_TOLERANCE,
    baselines: dict[str, float] | None = None,
    path: Path | None = None,
):
    """Professional model-selection visualization for GitHub README.

    Left: All Random Forest candidates, with only the winner highlighted.
    Right: Naive baseline RMSE comparison.
    """

    import numpy as np
    from matplotlib.figure import Figure

    # --------------------------------------------------
    # Color palette
    # --------------------------------------------------
    BLUE = "#2563EB"
    GREY = "#A3ACBA"
    BLACK = "#171717"
    TEXT = "#202938"
    MUTED = "#64748B"
    GRID = "#E5E7EB"

    df = table.copy()

    winners = df.loc[df["winner"].astype(bool)]

    if len(winners) != 1:
        raise ValueError("Expected exactly one selected winner.")

    winner = winners.iloc[0]

    best_rmse = df["val_rmse"].min()
    threshold = best_rmse * (1 + tolerance)

    # --------------------------------------------------
    # Figure layout
    # --------------------------------------------------
    fig = Figure(
        figsize=(14, 6.5),
        facecolor="white",
    )

    ax, ax_base = fig.subplots(
        1,
        2,
        gridspec_kw={
            "width_ratios": [3.2, 1.25],
            "wspace": 0.44,
        },
    )

    # --------------------------------------------------
    # LEFT PANEL: Random Forest candidates
    # --------------------------------------------------
    other = df.loc[~df["winner"].astype(bool)]

    ax.scatter(
        other["model_mb"],
        other["val_rmse"],
        s=95,
        color=GREY,
        alpha=0.85,
        edgecolors="white",
        linewidths=1.4,
        label="Random Forest candidates",
        zorder=3,
    )

    # Highlight selected model
    ax.scatter(
        [winner["model_mb"]],
        [winner["val_rmse"]],
        s=245,
        color=BLUE,
        edgecolors="white",
        linewidths=2.2,
        label="Selected serving model",
        zorder=6,
    )

    # Winner annotation below the blue dot
    winner_label = (
        f"{winner['candidate']}\nSize: {winner['model_mb']:.1f} MiB\nRMSE: {winner['val_rmse']:.4f} min"
    )

    ax.annotate(
        winner_label,
        xy=(
            winner["model_mb"],
            winner["val_rmse"],
        ),
        xytext=(12, -20),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=BLUE,
        ha="left",
        va="top",
        linespacing=1.5,
        zorder=7,
    )

    # Selection threshold
    ax.axhline(
        threshold,
        color=MUTED,
        linestyle="--",
        linewidth=1.3,
        label=f"RMSE tolerance (+{tolerance:.0%})",
        zorder=2,
    )

    # Axis limits
    lower = min(df["val_rmse"].min(), threshold)
    upper = max(df["val_rmse"].max(), threshold)

    margin = max(
        (upper - lower) * 0.20,
        0.025,
    )

    ax.set_ylim(
        lower - margin,
        upper + margin,
    )

    ax.set_xlim(
        0,
        df["model_mb"].max() * 1.08,
    )

    ax.set_title(
        "Random Forest Candidates",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        pad=16,
    )

    ax.set_xlabel(
        "Serialized Model Size (MiB)",
        fontsize=11,
        color=TEXT,
        labelpad=12,
    )

    ax.set_ylabel(
        "Validation RMSE (minutes)",
        fontsize=11,
        color=TEXT,
        labelpad=12,
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=False,
    )

    # --------------------------------------------------
    # RIGHT PANEL: Naive baselines
    # --------------------------------------------------
    baseline_items = [
        (str(name), float(value)) for name, value in (baselines or {}).items() if np.isfinite(float(value))
    ]

    if baseline_items:
        labels = [name for name, _ in baseline_items]
        values = [value for _, value in baseline_items]
        positions = np.arange(len(labels))

        ax_base.scatter(
            values,
            positions,
            s=155,
            marker="D",
            color=BLACK,
            edgecolors="white",
            linewidths=1.3,
            zorder=5,
        )

        for i, value in enumerate(values):
            ax_base.annotate(
                f"{value:.3f}",
                xy=(value, i),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=BLACK,
                va="center",
            )

        ax_base.set_yticks(positions)
        ax_base.set_yticklabels(
            labels,
            fontsize=10,
            color=TEXT,
        )

        ax_base.invert_yaxis()

        ax_base.set_xlim(
            0,
            max(values) * 1.30,
        )

        ax_base.set_ylim(
            len(labels) - 0.5,
            -0.5,
        )

    else:
        ax_base.text(
            0.5,
            0.5,
            "No baseline data available",
            transform=ax_base.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=11,
        )

        ax_base.set_yticks([])

    ax_base.set_title(
        "Naive Baselines",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        pad=16,
    )

    ax_base.set_xlabel(
        "Validation RMSE (minutes)",
        fontsize=11,
        color=TEXT,
        labelpad=12,
    )

    # --------------------------------------------------
    # Shared professional styling
    # --------------------------------------------------
    for axis in (ax, ax_base):
        axis.set_facecolor("white")

        axis.grid(
            axis="x",
            color=GRID,
            linewidth=0.8,
            alpha=0.85,
        )

        axis.set_axisbelow(True)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        axis.spines["left"].set_color(GRID)
        axis.spines["bottom"].set_color(GRID)

        axis.tick_params(
            axis="both",
            labelsize=9,
            colors=MUTED,
            length=0,
            pad=7,
        )

    # --------------------------------------------------
    # Main title and explanation
    # --------------------------------------------------
    fig.suptitle(
        "Model Selection | Accuracy vs. Efficiency",
        x=0.07,
        y=0.99,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
    )

    fig.text(
        0.07,
        0.015,
        "Lower RMSE is better  |  "
        "Selection favors the smallest model within "
        f"{tolerance:.0%} of the best validation RMSE  |  "
        "Baselines are accuracy references",
        fontsize=9,
        color=MUTED,
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.19,
        left=0.07,
        right=0.94,
    )

    # --------------------------------------------------
    # Save high-quality images for README
    # --------------------------------------------------
    if path is not None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            path.with_suffix(".svg"),
            bbox_inches="tight",
            facecolor="white",
        )

    return fig


# ------------------------------------------------------------
# 5. Final selection pipeline
# ------------------------------------------------------------


def _slug(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def run_selection(
    tolerance: float = RMSE_TOLERANCE,
    out_dir: Path = OUTPUT_DIR,
) -> dict:
    """Select using validation, then test and register winner."""

    import mlflow

    from src.baselines import read_baseline_runs
    from src.bundle import period_of
    from src.experiment import (
        MODEL_NAME,
        TRACKING_URI,
        log_model_artifacts,
    )
    from src.features import build_features, build_zone_xy
    from src.tracking import alias_version
    from src.train import (
        load_splits,
        period_params,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    champion_before = alias_version(
        MODEL_NAME,
        "champion",
    )

    # --------------------------------------------------------
    # STEP 1: Select using validation results only
    # --------------------------------------------------------

    table = apply_rule(
        collect_candidates(EXPERIMENT),
        tolerance,
    )

    winner = table.iloc[0]
    params = candidate_params(winner)

    # The RF baseline is scored on test next to the winner: fail now, before any refit, if its
    # latest run cannot be used (e.g. an older run without val_rmse or model_mb).
    rf_rows = table.loc[table["candidate"] == RF_BASELINE_RUN]
    if rf_rows.empty:
        raise LookupError(
            f"The latest '{RF_BASELINE_RUN}' run has no val_rmse or model_mb, so it cannot be "
            "compared on test. Run `uv run nyc-taxi train` again, then `uv run nyc-taxi select`."
        )
    rf_row = rf_rows.iloc[0]

    print(
        f"{int(table['tied'].sum())} of "
        f"{len(table)} candidates within "
        f"{tolerance:.0%} of best validation RMSE"
    )

    print(
        f"Winner: {winner['candidate']} | "
        f"Validation RMSE: {winner['val_rmse']:.4f} | "
        f"Model size: {winner['model_mb']:.1f} MiB"
    )

    # --------------------------------------------------------
    # STEP 2: Load data AFTER winner is selected
    # --------------------------------------------------------

    splits = load_splits()

    zone_xy = build_zone_xy(ROOT / "data")

    X_sample = build_features(
        splits.sample,
        zone_xy,
    )

    y_sample = splits.sample["duration"]

    X_test = build_features(
        splits.test,
        zone_xy,
    )

    # --------------------------------------------------------
    # STEP 3: Refit winning configuration
    # --------------------------------------------------------

    model = ev.fit_rf(
        params,
        X_sample,
        y_sample,
    )

    model_mb = ev.model_size_mb(model)

    # --------------------------------------------------------
    # STEP 4: Evaluate selected model on test
    # --------------------------------------------------------

    test_predictions = model.predict(X_test)

    test_metrics = ev.score_predictions(
        "test",
        splits.train,
        splits.test,
        test_predictions,
    )

    # --------------------------------------------------------
    # STEP 5: Evaluate reference baselines on same test
    # --------------------------------------------------------

    baseline_runs, _ = read_baseline_runs(EXPERIMENT)

    rows = []

    for run_name, (predict, _) in NAIVE_BASELINES.items():
        label = NAIVE_LABELS[run_name]

        predictions = predict(
            splits.train,
            splits.test,
        )

        scores = ev.score_predictions(
            "test",
            splits.train,
            splits.test,
            predictions,
        )

        val_rmse = (
            baseline_runs[label].data.metrics.get(
                "val_rmse",
                np.nan,
            )
            if label in baseline_runs
            else np.nan
        )

        rows.append(
            {
                "model": label,
                "val_rmse": val_rmse,
                **scores,
            }
        )

    # Random Forest baseline (rf_row was checked in step 1)
    if winner["candidate"] == RF_BASELINE_RUN:
        rf_model = model
    else:
        rf_model = ev.fit_rf(
            candidate_params(rf_row),
            X_sample,
            y_sample,
        )

    rf_predictions = rf_model.predict(X_test)

    rf_scores = ev.score_predictions(
        "test",
        splits.train,
        splits.test,
        rf_predictions,
    )

    rows.append(
        {
            "model": "RF baseline",
            "val_rmse": rf_row["val_rmse"],
            **rf_scores,
        }
    )

    if rf_model is not model:
        del rf_model

    winner_label = f"Winner ({winner['candidate']})"

    rows.append(
        {
            "model": winner_label,
            "val_rmse": winner["val_rmse"],
            **test_metrics,
        }
    )

    comparison = pd.DataFrame(rows).set_index("model")

    # --------------------------------------------------------
    # STEP 6: Export tables and professional chart
    # --------------------------------------------------------

    out_dir = Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        name: out_dir / name
        for name in (
            CANDIDATES_FILE,
            TEST_FILE,
            RULE_FILE,
            CHART_FILE,
        )
    }

    table.to_csv(
        paths[CANDIDATES_FILE],
        index=False,
    )

    comparison.to_csv(
        paths[TEST_FILE],
    )

    paths[RULE_FILE].write_text(
        rule_text(tolerance),
        encoding="utf-8",
    )

    naive_val = {
        label: float(row["val_rmse"])
        for label, row in comparison.iterrows()
        if label in NAIVE_LABELS.values()
    }

    plot_rmse_vs_size(
        table,
        tolerance,
        naive_val,
        path=paths[CHART_FILE],
    )

    # --------------------------------------------------------
    # STEP 7: Log and register without changing champion
    # --------------------------------------------------------

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

        for label, row in comparison.drop(index=winner_label).iterrows():
            metrics[f"test_{_slug(str(label))}_rmse"] = float(row["test_rmse"])

            metrics[f"test_{_slug(str(label))}_mae"] = float(row["test_mae"])

        mlflow.log_metrics({key: value for key, value in metrics.items() if np.isfinite(value)})

        mlflow.set_tag(
            "winner",
            winner["candidate"],
        )

        for artifact_path in paths.values():
            mlflow.log_artifact(str(artifact_path))

        # Also log vector chart
        svg_path = paths[CHART_FILE].with_suffix(".svg")

        if svg_path.exists():
            mlflow.log_artifact(str(svg_path))

        version = log_model_artifacts(
            model,
            zone_xy=zone_xy,
            register_name=MODEL_NAME,
            X=X_sample,
            training_period=period_of(splits.sample),
        )

        mlflow.set_tag(
            "registered_version",
            str(version),
        )

        champion_after = alias_version(
            MODEL_NAME,
            "champion",
        )

        if champion_after != champion_before:
            raise RuntimeError("Champion alias changed unexpectedly.")

    print(
        f"Winner: {winner['candidate']} | "
        f"Validation RMSE: {winner['val_rmse']:.4f} | "
        f"Test RMSE: {test_metrics['test_rmse']:.4f} | "
        f"Size: {model_mb:.1f} MiB | "
        f"Version: {version}"
    )

    print(f"Champion alias unchanged: {champion_after}")

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
