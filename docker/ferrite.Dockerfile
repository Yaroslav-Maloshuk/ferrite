FROM rust:1.98-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    libclang-dev clang cmake pkg-config libssl-dev libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --features bench --bins \
    && cp target/release/ferrite /usr/local/bin/ferrite \
    && cp target/release/ferrite-bench /usr/local/bin/ferrite-bench

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/ferrite /usr/local/bin/ferrite
COPY --from=builder /usr/local/bin/ferrite-bench /usr/local/bin/ferrite-bench
ENV FERRITE_DATA_DIR=/data FERRITE_MODEL_DIR=/models FERRITE_LANCE_URI=/data/lance FERRITE_PORT=8080
VOLUME ["/data"]
EXPOSE 8080
CMD ["ferrite", "serve"]
