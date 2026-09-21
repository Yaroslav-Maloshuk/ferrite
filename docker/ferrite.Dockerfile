FROM rust:1.98-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    libclang-dev clang cmake pkg-config libssl-dev libprotobuf-dev protobuf-compiler ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --features bench --bins \
    && cp target/release/ferrite /usr/local/bin/ferrite \
    && cp target/release/ferrite-bench /usr/local/bin/ferrite-bench
# Bake the embedding model into the image so the image is fully air-gapped at
# runtime: the checksum below is recorded in `/models/.manifest.json` by
# prefetch, and runtime verifies it against FERRITE_MODEL_SHA256 on every boot.
ARG FERRITE_MODEL_SHA256=53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
ENV FERRITE_MODEL_DIR=/models FERRITE_MODEL_SHA256=$FERRITE_MODEL_SHA256
RUN /usr/local/bin/ferrite prefetch && ls -1 /models
# Sanity check: the pinned hash must match the downloaded weights.
RUN test "$(sha256sum /models/model.safetensors | cut -d' ' -f1)" = "$FERRITE_MODEL_SHA256"

FROM debian:trixie-slim
ARG FERRITE_MODEL_SHA256=53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/ferrite /usr/local/bin/ferrite
COPY --from=builder /usr/local/bin/ferrite-bench /usr/local/bin/ferrite-bench
COPY --from=builder /models /models
ENV FERRITE_DATA_DIR=/data FERRITE_MODEL_DIR=/models FERRITE_LANCE_URI=/data/lance FERRITE_PORT=8080 \
    HF_HUB_OFFLINE=1 FERRITE_MODEL_SHA256=$FERRITE_MODEL_SHA256
VOLUME ["/data"]
EXPOSE 8080
CMD ["ferrite", "serve"]
