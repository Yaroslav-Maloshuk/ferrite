# Ferrite

Rust-native embedding (Candle) + retrieval (LanceDB) service, built as a drop-in
replacement for a Python/LangChain embedding + retrieval pipeline.

The service embeds text with `sentence-transformers/all-MiniLM-L6-v2` (384 dims,
pinned to `refs/pr/21`) using Candle, stores vectors in embedded LanceDB, and
serves the same HTTP API as the reference FastAPI/LangChain/Chroma baseline so
either can be benchmarked behind an identical workload.

## Quickstart

### Local

```bash
# fetch the model, then serve
cargo run --release --bin ferrite -- prefetch
FERRITE_LANCE_URI=data/lance cargo run --release --bin ferrite -- serve
```

```bash
# ingest and search over HTTP
curl -X POST localhost:8080/v1/ingest -H 'content-type: application/json' \
  -d '{"items":[{"id":"1","text":"the cat sits outside"},{"id":"2","text":"how to cook pasta"}]}'
curl -X POST localhost:8080/v1/search -H 'content-type: application/json' \
  -d '{"query":"cat on a couch","top_k":2}'
```

### Docker (one command)

```bash
docker compose up --build          # ferrite :8080 + baseline :8081
docker compose run --rm bench      # benchmarks both, prints the comparison table
```

The `bench` service generates the deterministic dataset, ingests it through each
service's HTTP ingest path, ramps concurrency, and writes reports + a comparison
to `data/reports/`.

## HTTP API

All routes are byte-compatible with the Python baseline.

| Route | Method | Body | Response |
| --- | --- | --- | --- |
| `/v1/embed` | POST | `{"texts":["..."]}` | `{embeddings, model, dim, latency_ms}` |
| `/v1/ingest` | POST | `{"items":[{"id":"1","text":"..."}]}` | `{"ingested": n}` |
| `/v1/search` | POST | `{"query":"...", "top_k":10}` | `{"results":[{id,text,score,distance}]}` |
| `/v1/health` | GET | — | `{"status":"ok"}` |
| `/v1/stats` | GET | — | `{rows, index, model, dim}` |

`top_k` is clamped to `[1, 1000]`; embed batches are capped at 256 texts.

## CLI

The `ferrite` binary (needs `--features bench` for the bench binary):

| Command | Purpose |
| --- | --- |
| `ferrite prefetch` | Download the model into `FERRITE_MODEL_DIR` (idempotent) |
| `ferrite serve [--port N]` | Run the HTTP service (binds `0.0.0.0`) |
| `ferrite ingest --file f.jsonl` | Bulk-ingest one `{"id","text"}` per line |
| `ferrite search --query q [--top-k n]` | Inspect the store |
| `ferrite stats` | Print store stats |
| `ferrite-bench dataset/run/compare/table` | Benchmark tooling (see `docs/benchmarks/benchmark-run-howto.md`) |

## Configuration

`FerriteConfig` is populated from environment variables (CLI flags win for
`serve --port`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `FERRITE_DATA_DIR` | `data` | Data root (reports, JSONL datasets) |
| `FERRITE_MODEL_DIR` | `data/models` | Where the embedding model is cached |
| `FERRITE_LANCE_URI` | `data/lance` | LanceDB table URI |
| `FERRITE_PORT` | `8080` | HTTP port |
| `FERRITE_INDEX` | `flat` | `flat` or `ivf_pq` |
| `FERRITE_TOP_K` | `10` | Default `top_k` for search |
| `FERRITE_BATCH_SIZE` | `32` | Model inference batch size |

## Tests

```bash
cargo test --features bench      # unit + integration tests
cargo clippy --features bench --all-targets -- -D warnings
cargo fmt --check
```

## Benchmarks

Methodology: a deterministic 25,000-question sample is drawn from the Quora
duplicates dataset (seed 42) and written to `quora_questions.jsonl`. The harness
ingests the first 5,000 documents through each service's HTTP ingest path, then
ramps concurrency `1, 2, 4, 8, cores, cores*2` over 10-second fixed windows of
1,000 probe queries (`top_k=10`), keeping the peak-throughput report. No tuning
or warmup separation: probes run while the service is under load.

Environment: macOS (Apple Silicon, 14 cores), native processes, Candle CPU path
(no `accelerate`), release build (`rust-version` 1.98, `lto=thin`).

```
| Metric | Ferrite | Python/LangChain | Speedup |
| --- | --- | --- | --- |
| P50 (ms) | 47.9 | 159.4 | 3.3x |
| P99 (ms) | 62.2 | 235.0 | 3.8x |
| P99.9 (ms) | 68.4 | 237.2 | 3.5x |
| Throughput RPS | 247.8 | 170.8 | 1.5x |
| RPS/core | 17.7 | 12.2 | 1.5x |
| Peak RSS (MB) | 20.0 | 21.0 | 1.1x |
FERRITE_ENV: os=macos cores=14 rust=0.1.0 container=false
BASELINE_ENV: os=macos cores=14 rust=0.1.0 container=false
```

Reproduce with `docs/benchmarks/benchmark-run-howto.md`. The same workload in
Docker (4-vCPU Linux containers: `cores=4 container=true`) measured Ferrite
p50 = 38.5 ms / 196 RPS vs baseline p50 = 40.1 ms / 183.2 RPS — with equally
shaped containers the gap squeezes and per-core throughput (`rps/core`
49.0 vs 45.8) is the cleaner signal.

Notes:
- `peak_rss_mb` is the benchmark harness process, not the server.
- Enabling `--features accelerate` (`candle-core/accelerate`, Apple
  Accelerate BLAS) can lower embedding latency on macOS; the numbers above are
  the baseline CPU build.

## Repository layout

```
src/                 library + bins (ferrite serve, ferrite-bench)
src/bench/           benchmark harness (dataset, runner, HTTP target, compare)
benchmark/python-baseline/  FastAPI/LangChain/Chroma reference (byte-compatible)
docker/               base image for ferrite
docker-compose.yml    ferrite + baseline + one-shot bench
docs/superpowers/     design spec + implementation plan
docs/benchmarks/      how to reproduce benchmark runs
```

## License

MIT — see [LICENSE](LICENSE).