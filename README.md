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
| `FERRITE_MODEL_REPO` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace repo to download when the cache is cold |
| `FERRITE_MODEL_REVISION` | `refs/pr/21` | Revision of `FERRITE_MODEL_REPO` to download |
| `FERRITE_MODEL_SHA256` | (unset) | Pin `model.safetensors` to this SHA-256; every load fails loudly on mismatch (air-gapped integrity) |
| `FERRITE_LANCE_URI` | `data/lance` | LanceDB table URI |
| `FERRITE_PORT` | `8080` | HTTP port |
| `FERRITE_INDEX` | `flat` | `flat` or `ivf_pq` |
| `FERRITE_IVF_PARTITIONS` | `16` | IVF partitions for `ivf_pq` index |
| `FERRITE_TOP_K` | `10` | Default `top_k` for search |
| `FERRITE_BATCH_SIZE` | `32` | Model inference batch size |
| `FERRITE_MAX_TEXT_BATCH` | `256` | Max texts per embed/ingest call |
| `FERRITE_API_KEY` | (unset) | Require this bearer key on every route except `/v1/health` |
| `FERRITE_TLS_CERT` / `FERRITE_TLS_KEY` | (unset) | PEM cert/key paths; serving switches to HTTPS when both are set |
| `HF_HOME` | `~/.cache/huggingface` | hf-hub cache root for model downloads (native, not Ferrite-specific) |
| `HF_HUB_OFFLINE` | (unset) | Set `1` to force offline mode (hf-hub refuses network) |

## Enterprise / air-gapped mode

- **Offline model:** `ferrite prefetch` downloads once and writes
  `data/models/.manifest.json` (per-file SHA-256). Every later
  load (*including `serve`*) re-verifies the hashes and **touches no
  network**. With `FERRITE_MODEL_SHA256` set, `model.safetensors` is pinned
  to that exact hash and mismatches abort startup — the integrity hook a
  security review will ask for. The Docker image bakes the model and runs with
  `HF_HUB_OFFLINE=1`.
- **Auth:** set `FERRITE_API_KEY` and send `Authorization: Bearer <key>` (or
  `X-API-Key: <key>`) on every route; `/v1/health` stays public for
  healthchecks. Keys are compared in constant time. `ferrite-bench` picks up
  the same key via its own `FERRITE_API_KEY` environment variable.
- **TLS:** set `FERRITE_TLS_CERT`/`FERRITE_TLS_KEY` (PEM) to serve HTTPS
  natively (rustls), no reverse proxy required.
- **Audit:** every request is logged with
  `method, uri, status, latency_ms, remote_ip, authed` at INFO level
  (`ferrite::http` target), giving a per-request audit trail for compliance.

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
ramps concurrency over 10-second fixed windows of 1,000 probe queries
(`top_k=10`) at 1, 2, 4, 8, ... (capped at `cores*2`), stopping at the first
run whose P99 breaches `P99@concurrency=1 × p99_saturation_factor` (default
`2.0`, flag `--p99-saturation-factor`), and keeping the peak-throughput report.
No tuning or warmup separation: probes run while the service is under load.

Search scores are cosine similarity (`distance = 1 - score`), matching the
Chroma baseline's `hnsw:space: cosine`, so result values are directly
comparable between the two services.

Environment: macOS (Apple Silicon, 14 cores), native processes, Candle CPU path
(no `accelerate`), release build (`rust-version` 1.98, `lto=thin`).

```
| Metric | Ferrite | Python/LangChain | Speedup |
| --- | --- | --- | --- |
| P50 (ms) | 17.5 | 14.5 | 0.8x |
| P99 (ms) | 25.6 | 16.0 | 0.6x |
| P99.9 (ms) | 32.7 | 16.1 | 0.5x |
| Throughput RPS | 196.0 | 138.0 | 1.4x |
| RPS/core | 14.0 | 9.9 | 1.4x |
| Peak RSS (MB) | 275.0 | 696.0 | 2.5x |
FERRITE_ENV: os=macos cores=14 rust=0.1.0 container=false
BASELINE_ENV: os=macos cores=14 rust=0.1.0 container=false
RSS_PROCESS: ferrite=pid:17694 baseline=pid:17695
```

Each service is measured at its own peak-throughput concurrency under the P99
saturation gate — ferrite peaked at `c=4`, the baseline at `c=2` (both breach
2x unloaded P99 at higher concurrency) — so the latency columns are not
directly comparable across services; throughput and RSS are the apples-to-apples
read. `peak_rss_mb` is the service process's own measured peak
(`RSS_PROCESS` shows the sampled pid / `self` for the in-process lib run).

Reproduce with `docs/benchmarks/benchmark-run-howto.md`. A Docker run (4-vCPU
Linux containers: `cores=4 container=true`) measured Ferrite p50 = 41.6 ms /
181.6 RPS vs baseline p50 = 40.2 ms / 182.4 RPS — both peeked at `c=8` under
the same saturation ramp, so the columns are comparable there, and the gap
squeezes to parity (`rps/core` 45.4 vs 45.6; container CPU contention caps
both). In Docker, `peak_rss_mb` reports the bench harness process (cross-
container `--target-pid` is unavailable); use `docker stats` for service RSS.

Notes:
- `peak_rss_mb` describes whichever process the `rss_process` field names. For
  HTTP targets the harness only reports its own RSS unless you pass
  `--target-pid <server-pid>`; then it reports the server's. Lib-target runs
  measure themselves (`self`). The table above was re-measured with
  `--target-pid`, so its RSS rows are the services' own measured peaks.
- Enabling `--features accelerate` (`candle-core/accelerate`, Apple
  Accelerate BLAS) can lower embedding latency on macOS; the numbers above are
  the baseline CPU build.

## Platforms

- Linux (x86_64/aarch64, glibc) — the Docker image and compose stack target this.
- macOS (Apple Silicon and Intel) — native; Apple Accelerate BLAS via
  `--features accelerate`.
- Windows (x86_64) — native MSVC toolchain; Docker is Linux-only.

Inference is CPU-only on every platform (`Device::Cpu`); there is no GPU/Metal
backend. `peak_rss_mb` uses a true per-process high-water mark on Linux
(`VmHWM`) and Windows (`PeakWorkingSetSize`); on macOS a *different* process is
sampled for its current RSS via `ps` (a lower bound), while the calling process
uses `getrusage`.

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