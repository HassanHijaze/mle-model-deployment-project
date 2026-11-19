# NYC Taxi Trip Duration Prediction

[![CI](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml/badge.svg)](https://github.com/HassanHijaze/mle-model-deployment-project/actions/workflows/ci.yml)

Predicts NYC yellow taxi trip duration from the pickup zone, drop-off zone and pickup time.
Random Forest · MLflow (PostgreSQL) · FastAPI · Docker Compose.

## Results

Time-based split on January 2025 trips: train Jan 1–20, validation Jan 21–25, test Jan 26–31.

| Model | Split | RMSE (min) |
|---|---|---|
| Baseline (500k sample) | validation | 4.522 |
| Optuna-tuned (500k sample) | validation | 4.485 |
| **Final** (tuned, all 2.7M Jan 1–25 trips) | **test** | **4.312** |

More data helped far more than tuning. Details: [notebooks/exploration.ipynb](notebooks/exploration.ipynb).

## Quick start

```bash
uv sync
cp .env.example .env && mkdir -p data
docker compose up -d                # MLflow: localhost:5002, API: localhost:8000
uv run python -m src.train          # trains and registers baseline + final model
```

In the MLflow UI, give a model version the alias **`champion`**, then run `curl -X POST http://localhost:8000/reload`.

> The final model needs more memory than Docker's default. If the API stops while loading it, serve the baseline or raise Docker's memory (`memory=12GB` in `.wslconfig` on Windows).

## API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"PULocationID": 161, "DOLocationID": 236, "pickup_datetime": "2025-01-27T08:30:00"}'
```

Endpoints: `/predict`, `/reload`, `/health`, and a web form at `/`. Zones are 1–265; time is NYC local time without a timezone.

## Tests

`uv run pytest` — tests needing MLflow or a `champion` model skip automatically. CI runs tests and builds the Docker image.

## Notes

- **No leakage:** `trip_distance` and fares are only known after a trip, so distance comes from zone centers.
- **One feature path:** training and the API share `src/features.py`.
- **Next steps:** a smaller final model, prediction ranges, more months of data.