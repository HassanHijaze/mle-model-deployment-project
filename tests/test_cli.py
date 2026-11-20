"""Unit tests for the command-line interface: argument parsing and step order.

The steps are mocked: no data, no MLflow, no training.
"""

import pytest

from src import cli

STEP_ORDER = ["data", "tune", "train", "select"]


def record_steps(monkeypatch, calls, fail_at=None):
    """Replace every step with a mock that records its name (and fails at fail_at)."""

    def make(name):
        def step():
            calls.append(name)
            if name == fail_at:
                raise RuntimeError(f"{name} failed")
            return {}

        return step

    monkeypatch.setattr(cli, "STEPS", {name: make(name) for name in STEP_ORDER})


@pytest.mark.parametrize("command", ["data", "tune", "train", "select", "all"])
def test_parser_accepts_each_subcommand(command):
    assert cli.build_parser().parse_args([command]).command == command


@pytest.mark.parametrize("argv", [[], ["deploy"], ["train", "--unknown"]])
def test_parser_rejects_missing_or_unknown_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_all_runs_data_tune_train_select_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "check_mlflow", lambda: "http://test")
    monkeypatch.setattr(cli, "print_next_steps", lambda results: None)
    record_steps(monkeypatch, calls)
    assert cli.main(["all"]) == 0
    assert calls == STEP_ORDER


def test_all_stops_on_the_first_failing_step(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "check_mlflow", lambda: "http://test")
    record_steps(monkeypatch, calls, fail_at="tune")
    with pytest.raises(RuntimeError, match="tune failed"):
        cli.main(["all"])
    assert calls == ["data", "tune"]  # train and select never start


@pytest.mark.parametrize("command", STEP_ORDER)
def test_single_command_runs_only_its_step(monkeypatch, command):
    calls = []
    monkeypatch.setattr(cli, "check_mlflow", lambda: "http://test")
    monkeypatch.setattr(cli, "print_next_steps", lambda results: None)
    record_steps(monkeypatch, calls)
    assert cli.main([command]) == 0
    assert calls == [command]


def test_all_stops_when_mlflow_is_unreachable(monkeypatch, capsys):
    calls = []

    def unreachable():
        raise cli.PipelineError(f"not reachable; {cli.MLFLOW_START_HINT}")

    monkeypatch.setattr(cli, "check_mlflow", unreachable)
    record_steps(monkeypatch, calls)
    assert cli.main(["all"]) == 1
    assert calls == []
    assert "docker compose up -d postgres mlflow" in capsys.readouterr().err
