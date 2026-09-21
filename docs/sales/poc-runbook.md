# PoC Runbook (Engineering Delivery Guide)

Internal/contractual runbook for a Ferrite PoC / migration engagement.
Skeleton for what is shipped to the client at handover.

## 0. Intake

- Capture: corpus size (rows, avg text length), required index (`flat` / `ivf_pq`),
  node spec (cores, RAM), target RPS / P99, whether air-gap is required, model pin.
- Convert to fixed price per `COMMERCIAL_OFFER.md`.

## 1. Staging

```bash
cargo build --release            # shipping binary
cargo build --release --features bench   # benchmark harness
```

Provision a VM matching the client's spec. Confirm Rust 1.98+ on the host used for builds,
or build inside the Docker image (docker/ferrite.Dockerfile).

## 2. Corpus intake

Expect JSONL, one `{"id","text"}` per line. Validate:

```bash
./target/release/ferrite-bench validate --file corpus.jsonl   # counts, dupes, max text
```

Normalize ids if needed (`doc-000001`-style) to keep insert-idempotency honest.

## 3. Build the store on the client box

```bash
FERRITE_LANCE_URI=/srv/ferrite/lance \
FERRITE_MODEL_DIR=/srv/ferrite/models \
./target/release/ferrite prefetch
./target/release/ferrite ingest --file corpus.jsonl
./target/release/ferrite stats
```

Record `rows`, `index`, `dim`, and wall-clock ingest time in the report. Re-running
`ingest` is a plain append (no PK constraint) — note this in the report.

## 4. Head-to-head benchmark

Run against each service with `--target-pid` so RSS measures the *service*, not the harness:

```bash
FERRITE_LANCE_URI=/srv/ferrite/lance ./target/release/ferrite serve --port 8080 &
./target/release/ferrite-bench run   --url http://localhost:8080 \
                                     --dataset corpus.jsonl --ramp --target-pid $SERVICE_PID
# …against the incumbent (Python/LangChain/Chroma) the same way…
./target/release/ferrite-bench compare ferrite-report.json incumbent-report.json
```

The harness picks up `FERRITE_API_KEY` automatically if the service requires auth
(`Authorization: Bearer <key>` attached to every request).

Outputs land in `data/reports/{ferrite,baseline,comparison}.json`

## 5. Air-gapped ship

- Model: `ferrite prefetch` writes `data/models/.manifest.json` (per-file SHA-256).
  Pinning: set `FERRITE_MODEL_SHA256=<sha256 of model.safetensors>`; any mismatch
  aborts startup loudly. Runtime does not touch the network while files are present.
- Docker: `docker build -f docker/ferrite.Dockerfile .` bakes the model into `/models`
  and runs with `HF_HUB_OFFLINE=1`; the build itself fails if the pinned hash doesn't match.
- Closed network: `docker save ferrite-ferrite | gzip > ferrite-image.tar.gz`,
  `docker load` on target (no registry required).

## 6. Handover checklist

- [ ] `comparison.json` (throughput / P99 / RSS for both pipelines, same concurrency)
- [ ] Rendered benchmark table
- [ ] `ferrite-image.tar.gz` or rebuilt image artifact
- [ ] Service verified: `/v1/health` 200, search returns hit against a seeded query
- [ ] If auth required: wrong key → 401, correct key → 200, `/v1/health` public
- [ ] Runbook copy + 30-day support contact

## 7. Operational notes (for the report)

- `serve` binds `0.0.0.0`; put it behind your own reverse proxy for fleet routing; native
  TLS (`FERRITE_TLS_CERT`/`FERRITE_TLS_KEY`) removes the need for a proxy for one node.
- Audit trail: INFO events with `method, uri, status, latency_ms, remote_ip, authed`
  (tracing target `ferrite::http`) — forward stdout to your log collector.
- Ramp semantics: concurrency doubles 1,2,4,… and stops once P99 exceeds
  P99@concurrency=1 × `p99_saturation_factor` (default 2.0).