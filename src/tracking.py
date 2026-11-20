"""Small MLflow helpers: find runs, read their files, log dataset info, read aliases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPERIMENT = "nyc-taxi-duration"
MISSING_RUN_HINT = "Run `uv run nyc-taxi all` first (MLflow server must be running)."


def latest_run(run_name: str, experiment: str = EXPERIMENT):
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
        raise LookupError(
            f"No finished '{run_name}' run in MLflow experiment '{experiment}'. {MISSING_RUN_HINT}"
        )
    return runs[0]


def child_runs(parent_run_id: str, experiment: str = EXPERIMENT) -> pd.DataFrame:
    """All child runs of a parent run as a DataFrame (columns params.*, metrics.*, tags.*)."""
    import mlflow

    return mlflow.search_runs(
        experiment_names=[experiment], filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )


def download_run_file(run_id: str, name: str) -> Path:
    """Download one artifact file of a run (not the whole run) and return its local path."""
    import mlflow

    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name))


def log_datasets(datasets: dict[str, pd.DataFrame], source: str) -> None:
    """Log dataset periods and sizes into the ACTIVE run (<context>_start, _end, _rows params)."""
    import mlflow

    for context, df in datasets.items():
        pickup = df["tpep_pickup_datetime"]
        mlflow.log_params(
            {
                f"{context}_start": f"{pickup.min():%Y-%m-%d}",
                f"{context}_end": f"{pickup.max():%Y-%m-%d}",
                f"{context}_rows": len(df),
            }
        )
    mlflow.set_tag("data_source", source)


def alias_version(model_name: str, alias: str) -> str | None:
    """Version the alias points to, or None if the alias (or model) does not exist."""
    from mlflow import MlflowClient

    try:
        return str(MlflowClient().get_model_version_by_alias(model_name, alias).version)
    except Exception:
        return None
