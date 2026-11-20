# NYC Taxi Trip Duration Prediction

[![CI](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml/badge.svg)](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml)

Predicts NYC yellow taxi trip duration from the pickup zone, drop-off zone and pickup time.
**Stack:** Random Forest · MLflow (PostgreSQL) · FastAPI · Docker Compose

## Results

January 2025 only, fixed time-based split on 1–60 minute trips:
**train Jan 1–20 · validation Jan 21–25 · test Jan 26–31.** Every model is trained on the training days
only, compared on validation, and the selected model is scored once on test.

| Step | MLflow run(s) | What it holds |
|---|---|---|
| `nyc-taxi train` | `baseline-overall-median`, `baseline-zone-pair-median`, `rf-baseline` | validation RMSE/MAE of the baselines |
| `nyc-taxi tune` | `optuna-tuning` (20 trial runs) | validation RMSE/MAE and model size of every trial |
| `nyc-taxi select` | `select` | the rule, the tied runs, the winner and the test comparison; registers the winner |

**Selection rule:** the lowest validation RMSE wins, but runs within 1% of the best count as tied and the
smallest model (`model_mb`) among them wins. The numbers are in the `select` run and in
[notebooks/model_serving_selection.ipynb](notebooks/model_serving_selection.ipynb); the data analysis is
in [notebooks/exploration.ipynb](notebooks/exploration.ipynb).

Latest `select` run (2026-11-20; 9 of 21 candidates tied within 1%), from the notebook outputs. Every
model learned from Jan 1–20 only; test = Jan 26–31, scored once:

| Model | Validation RMSE | Test RMSE | Test MAE | Size |
|---|---:|---:|---:|---:|
| Overall median | 9.9949 | 10.1225 | 6.9366 | – |
| Zone-pair median | 5.4437 | 5.4237 | 3.6200 | – |
| RF baseline | 4.4771 | 4.4837 | 3.1329 | – |
| **Winner: trial-1** | **4.4435** | **4.4415** | **3.1041** | **112.4 MiB** |

RMSE and MAE in minutes. trial-1 is 0.34% above the best validation RMSE (inside the 1% tolerance) and
the smallest of the tied models. It is registered as version 8 of `nyc-taxi-duration-rf`.

Only the winner is registered (as a version of `nyc-taxi-duration-rf`, without an alias). Serve it by
giving it the `champion` alias: [docs/operations.md](docs/operations.md).

## Quick Start

```bash
uv sync && cp .env.example .env    # once
docker compose up -d postgres mlflow
uv run nyc-taxi all                # trains, tunes, selects and registers the winner
```

`all` runs `data` → `train` → `tune` → `select` (also available one by one); the winner is registered without an alias.
Check a version with `uv run nyc-taxi verify <version>`, set `champion` in the MLflow UI (localhost:5002),
then start the API and load the champion:

```bash
docker compose up -d api
export $(grep ADMIN_TOKEN .env)    # the token for /reload, from .env
curl -X POST http://localhost:8000/reload -H "X-Admin-Token: $ADMIN_TOKEN"
```

The notebooks only read the results from MLflow.

## API

| Endpoint | Description |
|---|---|
| `POST /predict` | Predicts trip duration; the response names the model version that answered |
| `POST /reload` | Loads the current `champion` as one bundle (model, zone table, feature schema), validates it, then swaps; needs `X-Admin-Token`. On failure the old model keeps serving |
| `GET /live` | The process is alive |
| `GET /ready` | A validated model is loaded (503 otherwise) |
| `GET /health` | Model, zone table and last reload status |
| `GET /metrics` | Prometheus metrics (see Monitoring) |
| `GET /` | Web form |

Every response carries `X-Request-ID` (yours, or a new one). Errors share one JSON format:
`{"code": ..., "message": ..., "request_id": ...}`. Logs are JSON lines. Promotion and rollback:
[docs/operations.md](docs/operations.md).

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}'
```

Zone IDs: 1–265. `pickup_datetime`: NYC local time, no timezone.

## Monitoring

The API exposes Prometheus metrics at `GET /metrics` (requests by route and status, latency histogram,
predictions and prediction errors, active model version, model load time, reload attempts and
failures). `docker compose up -d prometheus` scrapes it every 15 s; query it in the Prometheus UI at
http://localhost:9090. Example queries:

```promql
# p95 latency of /predict (seconds)
histogram_quantile(0.95, sum by (le) (rate(taxi_api_request_duration_seconds_bucket{route="/predict"}[5m])))
# share of requests answered with 5xx
sum(rate(taxi_api_requests_total{status=~"5.."}[5m])) / sum(rate(taxi_api_requests_total[5m]))
# predictions per second, by model version
sum by (model_version) (rate(taxi_api_predictions_total[5m]))
# model being served, and failed reloads in the last day
taxi_api_model_info
increase(taxi_api_reload_failures_total[1d])
```

Load test (Locust) and results template: [docs/benchmarks.md](docs/benchmarks.md).

## Tests

| What | Command | Locally | CI job |
|---|---|---|---|
| Format and lint | `uv run ruff format --check .` · `uv run ruff check .` | yes | `lint` |
| Types (src/, app/) | `uv run mypy` | yes | `lint` |
| Dependency vulnerabilities | `uv export --frozen --all-groups --no-emit-project --format requirements-txt -o req.txt` then `uv run pip-audit -r req.txt --require-hashes --disable-pip` | yes | `security` |
| Unit and API tests (fake models, no MLflow) | `uv run pytest` | yes | `tests` |
| Model and integration tests (real champion) | `uv run pytest` with MLflow and a champion | yes | skipped (no MLflow) |
| Docker images build | `docker build .` | yes | `docker` |
| End-to-end: compose stack, promote, upgrade, reject a broken version | `uv run pytest -m e2e tests/e2e` | yes (needs Docker) | `e2e` |
| Load test (1, 10, 50 users) | `uv run python tests/load/run_load_test.py` | yes (API running) | — |

The end-to-end test starts its own throwaway stack (compose project `nyc-taxi-e2e`, other ports), registers tiny
synthetic models and always removes its containers and volumes; it never touches the development registry.

## Design Notes

- **No leakage:** `trip_distance` and fares are only known after a trip; distance is derived from zone centers.
- **One feature path:** training and the API share `src/features.py`.
- **Zone map versioned with the model:** training logs the zone table (`zone_xy.csv`) in the same MLflow run; the API loads it with the champion, so no download at startup.
- **Manual promotion:** new versions are registered without an alias; `champion` is set by hand.

## Project Questions

**1. What is the RMSE of the final model?**
**4.4415 minutes** on the test days (Jan 26–31) for the selected model (trial-1), computed once by
`nyc-taxi select` (`test_rmse` in the `select` run; [notebooks/model_serving_selection.ipynb](notebooks/model_serving_selection.ipynb), section 5).

**2. What would you do differently with more time?**
- **Add new information:** live traffic, exact pickup and drop-off locations or event data, since zones and time alone limit how precise predictions can be.
- **Use more history:** train on several months to cover seasons, weather and holidays.
- **Train on all training trips:** every Random Forest is fitted on a 500k sample of the training days.
- **Search for smaller models directly:** the selection already picks the smallest model within 1% of the best RMSE;
  adding size to the Optuna objective (or limiting depth and leaves) could find even smaller ones.
- **Return prediction ranges:** derived from the spread of the trees, most useful for airport trips.