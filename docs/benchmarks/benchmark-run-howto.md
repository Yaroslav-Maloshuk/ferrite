# Running benchmarks

Ferrite ships a deterministic benchmark harness that measures the same workload
against Ferrite (Candle + LanceDB) and the reference Python baseline
(FastAPI + LangChain + Chroma), then renders a comparison. Both services expose
the same HTTP API; the harness drives them over HTTP so the real ingest and
search paths are exercised.

## Workload

- Dataset: a seed-42 deterministic sample of the Quora duplicates dataset
  (25,000 questions) written as one question per line to `quora_questions.jsonl`
  (`ferrite-bench dataset`). The sample is byte-identical between runs and
  between the two services.
- Measure: ingest the first `n_docs` documents through each service's
  `POST /v1/ingest` (batched at 256 items to respect the embed cap), then run
  `n_queries` probe searches (`POST /v1/search`, `top_k=10`) in fixed `window`
  windows, ramping concurrency `1, 2, 4, 8, ...` up to `cores*2`. The ramp stops
  at the first run whose P99 breaches `P99@concurrency=1 * --p99-saturation-factor`
  (default `2.0`), and the peak-throughput run is reported.

## Native (no Docker)

Prereqs: a Rust 1.98+ toolchain, the model prefetched (`ferrite prefetch`), and
the Python baseline venv (`benchmark/python-baseline/.venv`, see its
`requirements.lock.txt`).

```bash
cargo build --release --features bench --bins

# 1. dataset (idempotent; re-uses a cached download)
./target/release/ferrite-bench dataset --out data/quora_questions.jsonl

# 2. services on non-conflicting ports
rm -rf /tmp/native-lance
FERRITE_LANCE_URI=/tmp/native-lance ./target/release/ferrite serve --port 8097 &
FERRITE_PID=$!
BASELINE_PERSIST_DIR=/tmp/native-chroma \
  benchmark/python-baseline/.venv/bin/uvicorn server:app \
  --app-dir benchmark/python-baseline --host 127.0.0.1 --port 8098 &
BASELINE_PID=$!

# 3. per-service benchmarks (lib mode is in-process, no server needed)
#    --target-pid makes peak_rss_mb describe the *service* process.
./target/release/ferrite-bench run --label ferrite-native-lib \
  --dataset data/quora_questions.jsonl --window-secs 10 \
  --n-docs 5000 --n-queries 1000 --top-k 10 --ramp \
  --out data/reports/ferrite-native-lib.json
./target/release/ferrite-bench run --label ferrite-native \
  --target-http http://127.0.0.1:8097 --target-pid "$FERRITE_PID" \
  --dataset data/quora_questions.jsonl \
  --window-secs 10 --n-docs 5000 --n-queries 1000 --top-k 10 --ramp \
  --out data/reports/ferrite-native.json
./target/release/ferrite-bench run --label baseline-native \
  --target-http http://127.0.0.1:8098 --target-pid "$BASELINE_PID" \
  --dataset data/quora_questions.jsonl \
  --window-secs 10 --n-docs 5000 --n-queries 1000 --top-k 10 --ramp \
  --out data/reports/baseline-native.json

# 4. compare + table
./target/release/ferrite-bench compare \
  --ours data/reports/ferrite-native.json \
  --baseline data/reports/baseline-native.json \
  --out data/reports/comparison-native.json
./target/release/ferrite-bench table --comparison data/reports/comparison-native.json
```

Notes:

- `ferrite serve` binds `--port` (it does not read `FERRITE_PORT`). Pick ports
  that do not collide with the Docker compose stack.
- `peak_rss_mb` reflects the process named in the report's `rss_process` field.
  Pass `--target-pid <pid>` for an HTTP target and the harness samples *that*
  process (`VmHWM` on Linux and `PeakWorkingSetSize` on Windows are true peaks;
  macOS uses current RSS via `ps`, a lower bound on the peak). This works on
  every platform Ferrite builds on. Without the flag, HTTP mode falls back to the harness
  process (`harness`); lib mode always measures itself (`self`). In Docker the
  bench container cannot read the service container's memory, so use
  `docker stats` for service RSS there.
- The macOS build uses the Candle CPU path. `--features accelerate` opts into
  Apple Accelerate BLAS via `candle-core/accelerate`; rebuild and re-measure
  to quantify the difference.
- If the target service runs with `FERRITE_API_KEY` set, export the same key
  before running the bench: `FERRITE_API_KEY=<key> ./target/release/ferrite-bench run ...`
  — the harness attaches `Authorization: Bearer <key>` to every request.

## Docker (one command)

```bash
docker compose up --build            # ferrite :8080 + baseline :8081, healthy
docker compose run --rm bench        # dataset -> both services -> compare -> table
```

The `bench` service builds the same image as `ferrite` and runs the full ramped
benchmark end-to-end, writing `data/reports/{ferrite,baseline,comparison}.json`
to the shared `./data` volume. Repeat runs are safe: the harness re-ingests with
`doc-N` ids as plain inserts (LanceDB `add` has no primary-key constraint), so
the stores accumulate rather than erroring. The image bakes the embedding model
at build time and runs with `HF_HUB_OFFLINE=1`, so the compose stack is fully
air-gapped after build.

Reported docker numbers were produced on 4-vCPU Linux containers
(`cores=4 container=true`); the native numbers above were produced on macOS with
14 cores (`container=false`). Environment is recorded per report in the `env`
field, so two runs are comparable only when their `env` matches.

## Reading a report

```json
{
  "label": "ferrite-native",
  "env": { "cores": 14, "os": "macos", "rust": "0.1.0", "container": false },
  "params": { "n_docs": 5000, "n_queries": 1000, "top_k": 10,
              "concurrency": 14, "window_secs": 10 },
  "metrics": { "p50_ms": 47.9, "p99_ms": 62.2, "p999_ms": 68.4,
               "rps": 247.8, "rps_per_core": 17.7, "peak_rss_mb": 20,
               "rss_process": "pid:12345" }
}
```