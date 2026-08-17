#!/usr/bin/env bash
# Generate Go gRPC stubs for Triton's inference service.
#
# The gateway talks to Triton over gRPC because decoupled streaming (one request, many
# responses) is only exposed there — the HTTP API has no equivalent, and decoupled mode
# is the whole reason this architecture works.
#
#   bash scripts/gen_triton_proto.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_RELEASE="${TRITON_RELEASE:-24.08}"
OUT="$REPO/gateway/internal/tritonpb"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

command -v protoc >/dev/null || {
  echo "installing protoc"; apt-get update -qq && apt-get install -y -qq protobuf-compiler; }

export PATH="$PATH:$(go env GOPATH)/bin"
command -v protoc-gen-go >/dev/null || \
  go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34.2
command -v protoc-gen-go-grpc >/dev/null || \
  go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1

# Pinned to the same release as the server: the protobuf schema is versioned with it.
BASE="https://raw.githubusercontent.com/triton-inference-server/common/r${TRITON_RELEASE}/protobuf"
mkdir -p "$TMP/protobuf"
for f in grpc_service.proto model_config.proto; do
  echo "fetching $f"
  curl -fsSL "$BASE/$f" -o "$TMP/protobuf/$f"
done

# The upstream protos carry no go_package option, so it is injected here rather than by
# editing vendored copies that would drift.
for f in "$TMP"/protobuf/*.proto; do
  grep -q 'option go_package' "$f" || \
    sed -i 's|^package inference;|package inference;\noption go_package = "./tritonpb";|' "$f"
done

mkdir -p "$OUT"
protoc -I "$TMP/protobuf" \
  --go_out="$OUT" --go_opt=paths=source_relative \
  --go-grpc_out="$OUT" --go-grpc_opt=paths=source_relative \
  "$TMP"/protobuf/*.proto

echo "generated:"
ls -la "$OUT"
