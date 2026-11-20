"""Unit test for the serving-selection charts with baseline reference lines (made-up numbers)."""

import pandas as pd

from src import serving_selection as ss


def test_baselines_drawn_as_lines_or_listed_when_far_above_the_candidates():
    ranked = pd.DataFrame({
        "candidate": ["tuned-25", "tuned-50"], "status": ["selected", "eligible"],
        "val_rmse": [4.40, 4.35], "size_mb": [300.0, 600.0], "latency_p95_ms": [2.0, 3.5],
    })
    baselines = {"Zone-pair median": 5.4, "RF baseline": 4.5, "Overall median": 10.0}
    fig = ss.plot_tradeoff(ranked, "size_mb", "size", "Accuracy vs size", ss.SelectionPolicy(),
                           size_budget=True, baselines=baselines)

    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Zone-pair median (5.400)" in labels
    assert "RF baseline (4.500)" in labels
    assert "Overall median (10.000, above chart)" in labels  # 10.0 > 1.5 x 4.40
    assert fig.axes[0].get_ylim()[1] < 10.0  # the far baseline does not stretch the axis
