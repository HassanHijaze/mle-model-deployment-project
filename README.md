# NYC Taxi Trip Duration Prediction

[![CI](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml/badge.svg)](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml)

Predicts NYC yellow taxi trip duration from the pickup zone, drop-off zone and pickup time.
**Stack:** Random Forest · MLflow (PostgreSQL) · FastAPI · Docker Compose

## Results

Time-based split on 1–60 minute trips: **train Jan 1–20 · validation Jan 21–25 · test Jan 26–31**

| Model | Val RMSE | Test RMSE |
|---|---:|---:|
| Overall median (naive) | 9.995 | 10.090 |
| Median per zone pair (naive) | 5.444 | 5.327 |
| Random Forest baseline | 4.509 | 4.509 |
| Optuna-tuned | 4.466 | 4.462 |
| **Final model** | –¹ | **4.312** |



**Takeaways**
- The final model is **19% better** than the zone-pair median and **57% better** than a single guess.
- **More data beat tuning:** tuning gained ~1%, training on all trips another ~3%.
- **Trade-off:** the final model is several GB; the baseline is 4% worse.

Full analysis: [notebooks/exploration.ipynb](notebooks/exploration.ipynb)

### Model serving options

Both registered models are versions of `nyc-taxi-duration-rf`. Serve one by assigning it the `champion` alias.
The final Jan 1–25 model (`rf-final-jan1-25`) is only scored, not logged: it is several GB.

| Model | Chosen by | Size |
|---|---|---|
| **Serving model** (`serving-selection`) | accuracy/size/RAM/latency rule, tested on February | under 700 MB |
| **Baseline** (`rf-baseline`) | 50-tree baseline on a 500k sample | small |

## Quick Start

```bash
uv sync && cp .env.example .env    # once
docker compose up -d postgres mlflow
uv run nyc-taxi all                # tunes, trains, selects and registers the models
```

`all` runs `data` → `tune` → `train` → `select` (also available one by one); models are registered without an alias.
Check a version with `uv run nyc-taxi verify <version>`, set `champion` in the MLflow UI (localhost:5002),
start the API with `docker compose up -d api`, then
`curl -X POST http://localhost:8000/reload -H "X-Admin-Token: $ADMIN_TOKEN"`. The notebooks only read the results from MLflow.

## API

| Endpoint | Description |
|---|---|
| `POST /predict` | Predicts trip duration; the response names the model version that answered |
| `POST /reload` | Loads the current `champion` as one bundle (model, zone table, feature schema), validates it, then swaps; needs `X-Admin-Token`. On failure the old model keeps serving |
| `GET /live` | The process is alive |
| `GET /ready` | A validated model is loaded (503 otherwise) |
| `GET /health` | Model, zone table and last reload status |
| `GET /` | Web form |

Every response carries `X-Request-ID` (yours, or a new one). Errors share one JSON format:
`{"code": ..., "message": ..., "request_id": ...}`. Logs are JSON lines. Promotion and rollback:
[docs/operations.md](docs/operations.md).

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}'
```

Zone IDs: 1–265. `pickup_datetime`: NYC local time, no timezone.

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

The end-to-end test starts its own throwaway stack (compose project `nyc-taxi-e2e`, other ports), registers tiny
synthetic models and always removes its containers and volumes; it never touches the development registry.

## Design Notes

- **No leakage:** `trip_distance` and fares are only known after a trip; distance is derived from zone centers.
- **One feature path:** training and the API share `src/features.py`.
- **Zone map versioned with the model:** training logs the zone table (`zone_xy.csv`) in the same MLflow run; the API loads it with the champion, so no download at startup.
- **Manual promotion:** new versions are registered without an alias; `champion` is set by hand.

## Project Questions

**1. What is the RMSE of the final model?**
**4.31 minutes** on the test set (680,825 trips, Jan 26–31), RMSLE 0.304.

**2. What would you do differently with more time?**
- **Add new information:** live traffic, exact pickup and drop-off locations or event data, since zones and time alone limit how precise predictions can be.
- **Use more history:** train on several months to cover seasons, weather and holidays.
- **Re-tune on full data:** hyperparameters were tuned on a 500k sample.
- **Reduce model size:** larger leaves or fewer trees for cheaper serving.
- **Return prediction ranges:** derived from the spread of the trees, most useful for airport trips.