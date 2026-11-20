# API Load Test Results

Locust benchmarks of the NYC Taxi Trip Duration Prediction API, running locally through Docker Compose with **MLflow model version 8**.

## How to Run

```bash
docker compose up -d postgres mlflow api prometheus
curl http://localhost:8000/ready

uv run python tests/load/run_load_test.py --users 1 10 --run-time 2m
```

Each scenario runs for two minutes. Locust saves its CSV results and a summary file in `tests/load/results/`.

## Test Environment

| Configuration | Value |
|---|---|
| Test date | November 20, 2026 |
| Operating system | Windows host |
| Load generator | Locust 2.47.0 |
| API deployment | Local Docker Compose |
| API URL | `http://localhost:8000` |
| Model version | 8 (`champion`) |
| Selected model | Random Forest, Optuna `trial-1` |
| Serialized model size | 112.4 MiB |
| Test duration | 2 minutes per scenario |

## Results

### Overall HTTP Traffic

Includes both `/predict` and `/ready` requests, as reported in the generated summary CSV.

| Users | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Error rate | Peak API RAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 109 | 0.93 | 77 | 80 | 82 | 0% | 627.6 MiB |
| 10 | 1,112 | 9.30 | 76 | 120 | 170 | 0% | 628.6 MiB |

### Prediction Endpoint Performance

These results include only `POST /predict`, excluding readiness checks.

| Users | Predictions | Prediction RPS | p50 | p95 | p99 | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 106 | 0.89 | 77 ms | 80 ms | 82 ms | 0% |
| 10 | 1,060 | 8.87 | 77 ms | 120 ms | 170 ms | 0% |


## Observations

- **Reliability:** Both completed scenarios reported zero failed requests.
- **Median latency:** Prediction p50 remained at 77 ms when increasing from 1 to 10 users.
- **Tail latency:** Prediction p95 increased from 80 ms to 120 ms.
- **Throughput:** Prediction throughput increased from 0.89 to 8.87 requests per second.
- **Memory:** Sampled peak API container memory increased from 627.6 MiB to 628.6 MiB.
- **Capacity:** These results characterize the configured Locust workload, not the API's maximum throughput.



**Raw results:** `tests/load/results/20261120-150927_summary.csv` .