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

<sub>Baseline and tuned: 500k-trip sample. Final: tuned settings on all 2.7M Jan 1–25 trips
(`n_estimators=200`, `max_depth=25`, `min_samples_leaf=8`, `max_features=0.5`).<br>
¹ Trained on the validation days; the same settings trained on Jan 1–20 scored 4.344.</sub>

**Takeaways**
- The final model is **19% better** than the zone-pair median and **57% better** than a single guess.
- **More data beat tuning:** tuning gained ~1%, training on all trips another ~3%.
- **Trade-off:** the final model is several GB; the baseline is 4% worse but small enough for default Docker memory.

Full analysis: [notebooks/exploration.ipynb](notebooks/exploration.ipynb)

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