"""The selection rule, candidate parsing and the RMSE-vs-size chart (made-up numbers, no MLflow)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import selection


def candidates(rows):
    """rows: (candidate, val_rmse, model_mb)."""
    return pd.DataFrame(
        [
            {"candidate": c, "run_id": f"id-{c}", "val_rmse": r, "val_mae": r / 2, "model_mb": mb}
            for c, r, mb in rows
        ]
    )


def test_lowest_rmse_wins_when_nothing_is_tied():
    table = selection.apply_rule(candidates([("a", 4.0, 500.0), ("b", 4.2, 10.0), ("c", 5.0, 1.0)]))
    assert table.loc[table["winner"], "candidate"].tolist() == ["a"]
    assert table["tied"].tolist() == [True, False, False]


def test_smallest_model_wins_among_runs_within_one_percent():
    # best 4.00 -> tied up to 4.04; b (4.03) is tied and smaller than a; c (4.05) is not tied
    table = selection.apply_rule(candidates([("a", 4.00, 900.0), ("b", 4.03, 120.0), ("c", 4.05, 5.0)]))
    assert table.iloc[0]["candidate"] == "b" and table.iloc[0]["winner"]
    assert dict(zip(table["candidate"], table["tied"], strict=True)) == {"b": True, "a": True, "c": False}
    assert table.loc[table["candidate"] == "b", "rmse_vs_best_pct"].iloc[0] == pytest.approx(0.75)


def test_same_size_tie_goes_to_the_lower_rmse():
    table = selection.apply_rule(candidates([("a", 4.02, 100.0), ("b", 4.00, 100.0)]))
    assert table.iloc[0]["candidate"] == "b"


def test_tolerance_is_a_parameter():
    rows = [("a", 4.00, 900.0), ("b", 4.10, 10.0)]
    assert selection.apply_rule(candidates(rows), tolerance=0.01).iloc[0]["candidate"] == "a"
    assert selection.apply_rule(candidates(rows), tolerance=0.05).iloc[0]["candidate"] == "b"


def test_candidate_params_parses_mlflow_text():
    row = {"n_estimators": "150", "max_depth": "30", "min_samples_leaf": "4", "max_features": "1.0"}
    params = selection.candidate_params(row)
    assert params == {"n_estimators": 150, "max_depth": 30, "min_samples_leaf": 4, "max_features": 1.0}
    assert isinstance(params["max_features"], float)  # 1.0 = all features; the integer 1 = one feature


def test_candidate_params_leaves_out_unset_params():
    # the RF baseline does not set max_features: scikit-learn's default must apply, as in its run
    row = {"n_estimators": "50", "max_depth": "20", "min_samples_leaf": "5", "max_features": np.nan}
    assert selection.candidate_params(row) == {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 5}


def test_collect_candidates_reads_rf_baseline_and_every_trial(monkeypatch):
    import src.tracking as tracking

    runs = {
        "rf-baseline": SimpleNamespace(
            info=SimpleNamespace(run_id="rf1"),
            data=SimpleNamespace(
                metrics={"val_rmse": 4.5, "val_mae": 2.9, "model_mb": 80.0},
                params={"n_estimators": "50", "max_depth": "20", "min_samples_leaf": "5"},
            ),
        ),
        "optuna-tuning": SimpleNamespace(info=SimpleNamespace(run_id="tune1"), data=SimpleNamespace()),
    }
    trials = pd.DataFrame(
        [
            {
                "run_id": "t0",
                "tags.mlflow.runName": "trial-0",
                "metrics.val_rmse": 4.4,
                "metrics.val_mae": 2.8,
                "metrics.model_mb": 300.0,
                "params.n_estimators": "100",
                "params.max_depth": "25",
                "params.min_samples_leaf": "8",
                "params.max_features": "0.5",
            },
            {  # an older trial without model_mb is left out
                "run_id": "t1",
                "tags.mlflow.runName": "trial-1",
                "metrics.val_rmse": 4.3,
                "metrics.val_mae": 2.7,
                "metrics.model_mb": np.nan,
            },
        ]
    )
    monkeypatch.setattr(tracking, "latest_run", lambda name, experiment=None: runs[name])
    monkeypatch.setattr(tracking, "child_runs", lambda run_id, experiment=None: trials)

    table = selection.collect_candidates()
    assert table["candidate"].tolist() == ["rf-baseline", "trial-0"]
    assert table.loc[1, "max_features"] == "0.5" and table.loc[0, "model_mb"] == 80.0


def test_chart_marks_winner_and_lists_far_baselines():
    table = selection.apply_rule(candidates([("rf-baseline", 4.5, 80.0), ("trial-0", 4.4, 300.0)]))
    fig = selection.plot_rmse_vs_size(table, baselines={"Zone-pair median": 5.4, "Overall median": 10.0})
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Zone-pair median (5.400)" in labels
    assert "Overall median (10.000, above chart)" in labels  # 10.0 > 1.5 x the worst candidate
    assert ax.get_xscale() == "log" and ax.get_ylim()[1] < 10.0


def test_rule_text_states_the_rule():
    text = selection.rule_text(0.01)
    assert "within 1% of the best" in text and "smallest model" in text and "once on test" in text
