#!/usr/bin/env bash
# Compile the tts_stream C++ backend and install it into the model repository.
#
#   bash scripts/build_backend.sh
#
# Runs on the GPU box (inside the Triton pod). Kept separate from provisioning so the
# C++ can be rebuilt in seconds during iteration.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/container_env.sh"
[ -f /workspace/env.sh ] && source /workspace/env.sh

SRC="$REPO/backends/tts_stream"
BUILD="$SRC/build"
# Triton looks for backends under <backend-directory>/<backend-name>/libtriton_<name>.so
DEST="${BACKEND_DIR:-/opt/tritonserver/backends}/tts_stream"
TRITON_SRC="${TRITON_SRC_DIR:-/opt/tts/tools/triton-src}"

command -v cmake >/dev/null || { echo "cmake missing — are you inside the Triton image?" >&2; exit 1; }
[ -d "$SRC" ] || { echo "no source at $SRC" >&2; exit 1; }
[ -d "$TRITON_SRC/backend/include" ] || {
  echo "Triton backend SDK not at $TRITON_SRC — run scripts/provision.sh first" >&2; exit 1; }

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_SRC_DIR="$TRITON_SRC"

cmake --build "$BUILD" -j"$(nproc)"

mkdir -p "$DEST"
cp "$BUILD/libtriton_tts_stream.so" "$DEST/"
echo
echo "installed -> $DEST/libtriton_tts_stream.so"
echo "undefined Triton symbols are resolved by the server at load time; this is expected:"
nm -D --undefined-only "$DEST/libtriton_tts_stream.so" | grep -c TRITON || true
echo
echo "load it with:"
echo "  curl -X POST localhost:8000/v2/repository/models/tts_stream/load"
