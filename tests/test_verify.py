"""`nyc-taxi verify <version>` and the bundle metadata, with MLflow mocked (runs in CI)."""

import json
import sys

import pytest

from src import bundle as bundle_lib
from src import cli
from src import verify as verify_mod
from tests.fakes import FakeMlflowClient, FakeModel, fake_zones, good_schema


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A fake registry: per version, the model and the artifact files of its run."""
    state = {"models": {}, "files": {}, "params": {}}

    def add(version, model=None, zones="default", schema="default", params=None):
        version = str(version)
        state["models"][version] = model if model is not None else FakeModel()
        files = {}
        if zones is not None:
            path = tmp_path / f"zones-{version}.csv"
            (fake_zones() if isinstance(zones, str) else zones).to_csv(path)
            files[bundle_lib.ZONE_FILE] = path
        if schema is not None:
            path = tmp_path / f"schema-{version}.json"
            path.write_text(json.dumps(good_schema() if isinstance(schema, str) else schema))
            files[bundle_lib.FEATURE_SCHEMA_FILE] = path
        state["files"][f"run-{version}"] = files
        state["params"][version] = params or {
            "bundle_training_start": "2025-01-01",
            "bundle_training_end": "2025-01-31",
        }

    def load_model(uri):
        model = state["models"][uri.rsplit("/", 1)[1]]
        if isinstance(model, Exception):
            raise model
        return model

    def download(run_id, artifact_path):
        try:
            return str(state["files"][run_id][artifact_path])
        except KeyError:
            raise OSError(f"artifact {artifact_path} not found in run {run_id}") from None

    monkeypatch.setattr(bundle_lib, "_client", lambda: FakeMlflowClient(state["params"]))
    monkeypatch.setattr(bundle_lib, "_load_model", load_model)
    monkeypatch.setattr(bundle_lib, "_download", download)
    return add


def test_verify_ok(registry, capsys):
    registry(3)
    assert verify_mod.verify_version("3") == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("OK")
    assert "training period: 2025-01-01 .. 2025-01-31" in out
    assert "161 -> 236" in out and "12.00 min" in out


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"schema": {"features": [{"name": "hour", "dtype": "int32"}]}},
            "feature schema does not match FEATURES",
        ),
        ({"zones": None}, "zone table zone_xy.csv missing"),
        ({"model": FakeModel(minutes=float("nan"))}, "not all finite"),
        ({"model": FakeModel(minutes=90.0)}, "outside 1-60 min"),
    ],
)
def test_verify_reports_the_exact_problem(registry, capsys, kwargs, expected):
    registry(4, **kwargs)
    assert verify_mod.verify_version("4") == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and expected in out


def test_verify_unknown_version(registry, capsys):
    assert verify_mod.verify_version("99") == 1
    assert "not found" in capsys.readouterr().out


def test_verify_corrupted_model(registry, capsys):
    registry(5, model=EOFError("unexpected end of file"))
    assert verify_mod.verify_version("5") == 1
    assert "could not be loaded" in capsys.readouterr().out


def test_verify_older_model_without_schema_uses_model_feature_names(registry, capsys):
    registry(6, schema=None)
    assert verify_mod.verify_version("6") == 0
    assert "feature_schema.json not logged" in capsys.readouterr().out


def test_verify_warns_about_library_version_differences(registry, capsys):
    registry(7, params={"lib_scikit_learn": "0.0.1"})
    assert verify_mod.verify_version("7") == 0
    assert "scikit-learn: trained with 0.0.1" in capsys.readouterr().out


def test_cli_verify_runs_in_a_fresh_process(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "check_mlflow", lambda: "http://test")
    monkeypatch.setattr(
        cli.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 1})()
    )
    assert cli.main(["verify", "12"]) == 1
    assert calls == [[sys.executable, "-m", "src.verify", "12"]]


@pytest.mark.parametrize("argv", [["verify"], ["verify", "abc"], ["verify", "0"]])
def test_cli_verify_needs_a_version_number(argv):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_log_bundle_metadata_writes_schema_period_and_versions(monkeypatch):
    import mlflow

    logged = {}
    monkeypatch.setattr(
        mlflow,
        "log_artifact",
        lambda path: logged.update(schema=json.loads(open(path, encoding="utf-8").read())),
    )
    monkeypatch.setattr(mlflow, "log_params", lambda params: logged.update(params=params))
    from src.features import FEATURES, build_features

    X = build_features(bundle_lib.reference_frame(), fake_zones())
    bundle_lib.log_bundle_metadata(X, ("2025-01-01", "2025-01-20"))
    assert [f["name"] for f in logged["schema"]["features"]] == FEATURES
    assert all(f["dtype"] for f in logged["schema"]["features"])
    assert logged["params"]["bundle_training_start"] == "2025-01-01"
    assert logged["params"]["bundle_training_end"] == "2025-01-20"
    assert {"lib_scikit_learn", "lib_pandas", "lib_numpy", "lib_mlflow"} <= set(logged["params"])
