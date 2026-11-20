"""Run the Locust load test headless at several user levels and save the results as CSV.

    uv run python tests/load/run_load_test.py                       # 1, 10 users, 2 min each
    uv run python tests/load/run_load_test.py --users 1 10  --run-time 5m --host http://localhost:8000

For each level, Locust writes tests/load/results/<stamp>_users<N>_stats.csv (and _stats_history,
_failures, _exceptions). This script adds tests/load/results/<stamp>_summary.csv with one row
per level: throughput, p50/p95/p99 latency, error rate and the API container's peak memory
(sampled with `docker stats` every 2 s, if the api runs in docker compose).

The API must be running with a champion loaded (`curl http://localhost:8000/ready`).
Copy the summary into docs/benchmarks.md. Locust rewrites its CSV every second, so requests that
finish in the last second of a level can be missing from it; use run times of a few minutes.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
UNITS = {"b": 1, "kib": 2**10, "mib": 2**20, "gib": 2**30, "kb": 1e3, "mb": 1e6, "gb": 1e9}


def api_container_id() -> str | None:
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "-q", "api"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def memory_mib(container: str) -> float | None:
    """Current memory use of a container in MiB, from `docker stats` ('412.5MiB / 2GiB')."""
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", out.stdout)
    if not match or match.group(2).lower() not in UNITS:
        return None
    return float(match.group(1)) * UNITS[match.group(2).lower()] / 2**20


class MemorySampler(threading.Thread):
    """Peak memory of the api container while a level runs."""

    def __init__(self, container: str | None, every_s: float = 2.0):
        super().__init__(daemon=True)
        self.container, self.every_s = container, every_s
        self.peak: float | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        while self.container and not self._stop_event.is_set():
            value = memory_mib(self.container)
            if value is not None:
                self.peak = value if self.peak is None else max(self.peak, value)
            self._stop_event.wait(self.every_s)

    def stop(self) -> float | None:
        self._stop_event.set()
        self.join(timeout=35)
        return self.peak


def aggregated_row(stats_csv: Path) -> dict:
    with stats_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Name"] == "Aggregated":
                return row
    raise RuntimeError(f"no Aggregated row in {stats_csv}")


def run_level(users: int, host: str, run_time: str, prefix: Path, container: str | None) -> dict:
    sampler = MemorySampler(container)
    sampler.start()
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(HERE / "locustfile.py"),
        "--headless",
        "--users",
        str(users),
        "--spawn-rate",
        str(users),
        "--run-time",
        run_time,
        "--host",
        host,
        "--csv",
        str(prefix),
        "--csv-full-history",
        "--only-summary",
    ]
    print(f"\n==> {users} user(s) for {run_time} against {host}", flush=True)
    returncode = subprocess.run(cmd, cwd=ROOT).returncode  # 1 = some requests failed; still reported
    peak = sampler.stop()
    row = aggregated_row(prefix.with_name(prefix.name + "_stats.csv"))
    requests, failures = int(row["Request Count"]), int(row["Failure Count"])
    return {
        "users": users,
        "run_time": run_time,
        "requests": requests,
        "throughput_rps": round(float(row["Requests/s"]), 2),
        "p50_ms": row["50%"],
        "p95_ms": row["95%"],
        "p99_ms": row["99%"],
        "error_rate_pct": round(100 * failures / requests, 3) if requests else "",
        "api_peak_memory_mib": round(peak, 1) if peak is not None else "",
        "locust_exit_code": returncode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--run-time", default="2m", help="per level, Locust format (30s, 2m, 1h)")
    args = parser.parse_args(argv)

    try:
        with urllib.request.urlopen(f"{args.host}/ready", timeout=10) as r:
            print("API ready:", r.read().decode())
    except Exception as exc:
        print(
            f"The API is not ready at {args.host} ({exc}). Start it and load a champion first.",
            file=sys.stderr,
        )
        return 1

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    container = api_container_id()
    if container is None:
        print("api container not found: memory will not be recorded")

    rows = []
    for users in args.users:
        rows.append(run_level(users, args.host, args.run_time, RESULTS / f"{stamp}_users{users}", container))
        time.sleep(5)  # let the API settle between levels

    summary = RESULTS / f"{stamp}_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary: {summary}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
