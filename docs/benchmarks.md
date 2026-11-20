# API load test results

Template: fill in every `<...>` after running the load test. Leave nothing estimated: every number
here must come from a run.

## How to run

```bash
docker compose up -d postgres mlflow api prometheus
curl http://localhost:8000/ready                       # a champion must be loaded
uv run python tests/load/run_load_test.py              # 1, 10 and 50 users, 2 min each
# options: --users 1 10 50  --run-time 5m  --host http://localhost:8000
```

Results land in `tests/load/results/` (git-ignored): Locust's CSV files per level and
`<timestamp>_summary.csv` with one row per level (the numbers for the table below). While it runs,
Prometheus (http://localhost:9090) shows the API's own view (see the README "Monitoring" section).

## Hardware

| | |
|---|---|
| Machine / CPU | `<model, cores / threads>` |
| RAM | `<GB>` |
| OS, Docker version | `<...>` |
| Load generator | `<same machine as the API? yes/no>` |

## Model

| | |
|---|---|
| Model version (champion) | `<version>` (`curl http://localhost:8000/ready`) |
| Winner run / params | `<from the MLflow run "select">` |
| Serialized size (`model_mb`) | `<MB>` |

## Settings

| | |
|---|---|
| API container limits | memory `<API_MEMORY_LIMIT>`, CPUs `<API_CPU_LIMIT>` |
| uvicorn workers | `<1 (default)>` |
| Locust | think time 0.5-1.5 s per user, ~95% `/predict`, ~5% `/ready`, random zones 1-263, January 2025 pickup times |
| Run time per level | `<e.g. 2m>` |
| Date of the run | `<YYYY-MM-DD>` |

## Results

| Users | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Error rate (%) | API peak memory (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| 10 | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |
| 50 | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

Latency is measured by Locust (client side, including HTTP); the API's own latency is in
`taxi_api_request_duration_seconds` in Prometheus.

## Observations

- `<where latency starts to grow, errors if any, memory behaviour, CPU-bound or not>`
