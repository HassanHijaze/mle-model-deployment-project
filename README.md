# NYC Taxi Trip Duration Prediction

[![CI](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml/badge.svg)](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml)

Predicts NYC yellow taxi trip duration from the pickup zone, drop-off zone and pickup time.
**Stack:** Random Forest · MLflow (PostgreSQL) · FastAPI · Docker Compose

## Results

Time-based split on 1–60 minute trips: **train Jan 1–20 · validation Jan 21–25 · test Jan 26–31**.
All models are scored on all validation trips (637,543) and all test trips (680,825).

| Model | Validation RMSE | Test RMSE |
|---|---|---|
| Always predict the overall median | 9.995 | 10.090 |
| Median per pickup/drop-off zone pair | 5.444 | 5.327 |
| Random Forest baseline (50 trees, 500k sample) | 4.509 | 4.509 |
| Optuna-tuned (20 trials, 500k sample) | 4.466 | 4.462 |
| **Final** (tuned, all 2.7M Jan 1–25 trips) | – | **4.312** |

"–": the final model is trained on the validation days, so it has no validation score.
The same settings trained on Jan 1–20 scored 4.344 on validation.

The final model is **19% better** than the zone-pair median and **57% better** than a single guess.
Tuned settings: `n_estimators=200`, `max_depth=25`, `min_samples_leaf=8`, `max_features=0.5`.
Full analysis: [notebooks/exploration.ipynb](notebooks/exploration.ipynb).

**Was the extra complexity worth it?** Tuning alone gave only ~1% (4.509 → 4.462 on test) and
made the model larger. Training on all trips gave a further ~3%, but the model grows to several GB.
For a production service, the small baseline (4% worse, a fraction of the size) is the more
practical choice; the large model is only worth it if every few seconds of accuracy matter.

### Model serving options

Both models are registered in MLflow as `nyc-taxi-duration-rf`. Serve one by assigning it the `champion` alias.

| Model | Test RMSE | Size | Memory |
|---|---|---|---|
| **Final** (`rf-final-jan1-25`) | 4.312 | several GB | requires increased Docker memory |
| **Baseline** (`rf-baseline`) | 4.509 | small | runs with Docker defaults |

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