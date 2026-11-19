# NYC Taxi Trip Duration: Model Deployment

Predicts how long a NYC Yellow Taxi trip will take (in minutes) from the **pickup zone, drop-off zone and pickup time**.
A Random Forest is trained on January 2025 trips, tracked and versioned in **MLflow**, and served through a **FastAPI** app. Everything runs locally with **Docker Compose**.


## Results

All splits are by time (1–60 minute trips): **train Jan 1–20, validation Jan 21–25, test Jan 26–31**.

| Model | Split | RMSE (min) |
|---|---|---|
| Baseline (50 trees, 500k sample) | validation | 4.522 |
| Optuna-tuned (20 trials, 500k sample) | validation | 4.485 |
| **Final** (tuned, retrained on all Jan 1–25 trips, 2.7M) | **test** | **4.312** |

Best settings: `n_estimators=200, max_depth=25, min_samples_leaf=8, max_features=0.5`.
The full analysis is in [notebooks/exploration.ipynb](notebooks/exploration.ipynb).

### Which model to serve

Training registers two versions of `nyc-taxi-duration-rf` in MLflow:

| Model (run name) | Accuracy | Size | Serving |
|---|---|---|---|
| **Final** (`rf-final-jan1-25`) | best (test RMSE 4.312) | several GB | needs more Docker memory (see below) |
| **Baseline** (`rf-baseline`) | lower (val RMSE 4.522) | small | works with Docker's default memory |

Choose one by giving it the `champion` alias in the MLflow UI.

## Project Questions

**1. What is the RMSE of your final model?**
**4.31 minutes** on the test set (all 680,825 trips from Jan 26–31, trips of 1–60 minutes), with an RMSLE of 0.304.

**2. What would you do differently with more time?**
- **New information instead of new features:** over 30 feature ideas built from zones and time (boroughs, bearing, weather, OSRM routes, zone speeds, ...) landed within noise. Real gains would need live traffic data, exact pickup and drop-off locations instead of zones, or event calendars.
- **More history:** train on several months, so holidays, weather and seasons appear in training, validation and test.
- **Re-tune on the full data:** the settings were tuned on a 500k sample and reused for 2.7M trips.
- **A smaller final model:** the final model is several GB. Larger leaves or fewer trees would make it cheaper to store and serve, at a small accuracy cost.
- **Prediction ranges:** return an interval (for example from the spread of the trees), especially for airport trips, where errors are largest.

## Quick Start

**Requirements:** Docker, [uv](https://docs.astral.sh/uv/) ≥ 0.13.5, Python 3.13, and several GB of free RAM and disk space for the final model.

**Memory for the final model:** loading it needs more memory than Docker Desktop gets by default
(about half your RAM on Windows/macOS). If the API keeps restarting while it loads the model,
either serve the baseline model instead, or give Docker more memory. On Windows, create
`%USERPROFILE%\.wslconfig` with:

```
[wsl2]
memory=12GB
```

then run `wsl --shutdown` in PowerShell and start Docker Desktop again.

```bash
# 1. Install dependencies and configure the environment
uv sync
cp .env.example .env          # then set your own Postgres password

# 2. Start Postgres, the MLflow server (localhost:5002) and the API (localhost:8000)
docker compose up -d

# 3. Train and register the models (downloads the data into data/ on the first run)
uv run python -m src.train
```

4. In the MLflow UI (http://localhost:5002), give the model version you want to serve the alias **`champion`**
   (the final model if Docker has enough memory, otherwise the baseline).
5. Load it into the running API:

```bash
curl -X POST http://localhost:8000/reload
```

## API

| Endpoint | Description |
|---|---|
| `GET /` | Simple web form for trying predictions |
| `POST /predict` | Predicts trip duration |
| `POST /reload` | Loads the current `champion` model without restarting |
| `GET /health` | Shows whether the model and zone map are loaded |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 132, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}'
```

Response format (the values are only an example):

```json
{"duration_minutes": 42.17, "model_version": "4"}
```

Zone IDs must be between 1 and 265. `pickup_datetime` is New York local time **without** a timezone.
Interactive docs are at http://localhost:8000/docs.

## Project Structure

```
app/          FastAPI app (main.py) and web form (static/)
src/          features.py, train.py, experiment.py (MLflow logging), tuning.py (Optuna)
notebooks/    Exploration, tuning and results summary
tests/        Feature, model and API tests
docker/       MLflow server image
```

## Tests

```bash
uv run pytest
```

Feature tests run without any setup. Model and API tests need the MLflow server running and a `champion` model; if either is missing, those tests are skipped. With the final model as champion, the tests also need several GB of RAM, since they load the model.

## Design Notes

- **No data leakage:** `trip_distance` and the fare columns are only known after a trip ends, so the model doesn't use them. Distance comes from the zone map instead (straight-line distance between zone centers).
- **One feature path:** training, the notebook and the API all build features with the same function (`src/features.py`), and a test checks that the API and the model give identical predictions.
- **Time-based split:** the model is evaluated on days it never saw, as it would be in production.
- **Tuning vs. more data:** Optuna improved validation RMSE by about 0.04 minutes, while training on all trips instead of a sample improved it by about 0.14. More data mattered far more than tuning.
- **Accuracy vs. serving cost:** the most accurate model is several GB, so the baseline is kept as a light option that runs with default Docker memory.
- **Human-in-the-loop promotion:** training registers new versions but never promotes them. The `champion` alias is set by hand, and `/reload` switches models with no downtime.
- **Graceful startup:** if the model or zone map is missing, the API still starts. `/predict` returns `503` and `/health` shows what is missing.
- **Limits:** the model only covers 1–60 minute trips and knows only zone-level locations. Real traffic and route data are not used.