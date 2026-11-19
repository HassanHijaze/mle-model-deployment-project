# NYC Taxi Trip Duration Prediction

[![CI](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml/badge.svg)](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml)

Predicts NYC yellow taxi trip duration from the pickup zone, drop-off zone and pickup time.
**Stack:** Random Forest · MLflow (PostgreSQL) · FastAPI · Docker Compose

## Results

Time-based split on 1–60 minute trips: **train Jan 1–20 · validation Jan 21–25 · test Jan 26–31**.

| Model | Evaluated on | RMSE (min) |
|---|---|---|
| Baseline (50 trees, 500k sample) | validation | 4.522 |
| Optuna-tuned (20 trials, 500k sample) | validation | 4.485 |
| **Final** (tuned, retrained on 2.7M Jan 1–25 trips) | **test** | **4.312** |

Tuned settings: `n_estimators=200`, `max_depth=25`, `min_samples_leaf=8`, `max_features=0.5`.
Full analysis: [notebooks/exploration.ipynb](notebooks/exploration.ipynb).

### Model serving options

Both models are registered in MLflow as `nyc-taxi-duration-rf`. Serve one by assigning it the `champion` alias.

| Model | RMSE | Size | Memory |
|---|---|---|---|
| **Final** (`rf-final-jan1-25`) | 4.312 (test) | several GB | requires increased Docker memory |
| **Baseline** (`rf-baseline`) | 4.522 (validation) | small | runs with Docker defaults |

## Quick Start

```bash
uv sync
cp .env.example .env && mkdir -p data
docker compose up -d              # MLflow: localhost:5002 · API: localhost:8000
uv run python -m src.train        # trains and registers both models
```

Assign the `champion` alias in the MLflow UI, then load it: `curl -X POST http://localhost:8000/reload`

> **Memory:** if the API stops while loading the final model, serve the baseline or increase Docker's memory (`memory=12GB` in `%USERPROFILE%\.wslconfig` on Windows).

## API

| Endpoint | Description |
|---|---|
| `POST /predict` | Predicts trip duration |
| `POST /reload` | Loads the current `champion` without restarting |
| `GET /health` | Model and zone map status |
| `GET /` | Web form |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}'
```

Zone IDs: 1–265. `pickup_datetime`: NYC local time, no timezone.

## Tests

```bash
uv run pytest
```

| Tests | Locally | In CI |
|---|---|---|
| Features, data filtering, time split | ✓ | ✓ |
| Model and API (skew, behavior, endpoints) | ✓ with MLflow running and a `champion` model | skipped (no MLflow server in CI) |

CI also builds the API Docker image on every push.

## Design Notes

- **No leakage:** `trip_distance` and fares are only known after a trip; distance is derived from zone centers.
- **One feature path:** training and the API share `src/features.py`.
- **Manual promotion:** new versions are registered without an alias; `champion` is set by hand.

## Project Questions

**1. What is the RMSE of the final model?**
**4.31 minutes** on the test set (680,825 trips, Jan 26–31), RMSLE 0.304.

**2. What would you do differently with more time?**
- **Add new information:** over 30 additional features from zones and time gave no measurable gain. Further improvement requires live traffic, exact locations or event data.
- **Use more history:** train on several months to cover seasons, weather and holidays.
- **Re-tune on full data:** hyperparameters were tuned on a 500k sample.
- **Reduce model size:** larger leaves or fewer trees for cheaper serving.
- **Return prediction ranges:** derived from the spread of the trees, most useful for airport trips.