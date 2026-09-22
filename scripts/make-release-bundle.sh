#!/usr/bin/env bash
# Assemble a release bundle from locally built, verified artifacts.
# Produces: dist/ferrite-<ver>/ (binaries + docs) and dist/ferrite-<ver>.tar.gz + SHASUMS.
# Optional --with-docker includes docker-saved linux images (large).
set -euo pipefail

VERSION="${1:-0.1.0}"
OUT="dist/ferrite-$VERSION"
mkdir -p "$OUT/bin/macos-arm64" "$OUT/bin/macos-x86_64" "$OUT/docs"

copy_platform() {
  local plat=$1 ferrite=$2 bench=$3
  [ -f "$ferrite" ] && [ -f "$bench" ] || { echo "missing binaries for $plat"; exit 1; }
  cp "$ferrite" "$OUT/bin/$plat/"
  cp "$bench" "$OUT/bin/$plat/"
}

copy_platform macos-arm64     target/release/ferrite                                     target/release/ferrite-bench
copy_platform macos-x86_64    target/x86_64-apple-darwin/release/ferrite                 target/x86_64-apple-darwin/release/ferrite-bench

for doc in LICENSE COMMERCIAL_LICENSE.md COMMERCIAL_OFFER.md README.md; do
  [ -f "$doc" ] && cp "$doc" "$OUT/docs/"
done
cp docs/sales/poc-runbook.md "$OUT/docs/"

if [ "${2:-}" = "--with-docker" ]; then
  echo "saving linux images (large)..."
  docker save ferrite-ent-test | gzip > "$OUT/linux-amd64.tar.gz"
  docker save ferrite-arm64-test | gzip > "$OUT/linux-arm64.tar.gz"
  echo "docker save done; verify via: docker load < linux-amd64.tar.gz && docker run -p 8080:8080 ferrite-ent-test"
fi

mkdir -p dist
tar -czf "$OUT.tar.gz" -C dist "ferrite-$VERSION"
(cd dist && shasum -a 256 "ferrite-$VERSION.tar.gz" > SHASUMS.txt)
echo "bundle: $OUT.tar.gz"
cat dist/SHASUMS.txt