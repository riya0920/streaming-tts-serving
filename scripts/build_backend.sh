#!/usr/bin/env bash
# Compile the tts_stream C++ backend against the Triton backend SDK and install it into
# the model repository. Runs inside the Triton container:
#
#   docker compose -f docker/docker-compose.yml exec triton build_backend.sh
#
# Kept out of the Dockerfile's RUN layers so the C++ can be recompiled in a live
# container during iteration without a 15 GB image rebuild.
set -euo pipefail

SRC=/workspace/backends/tts_stream
BUILD=${SRC}/build
DEST=/models/tts_stream/1

command -v cmake >/dev/null || { echo "cmake missing — are you inside the tts-triton image?" >&2; exit 1; }
[ -d "$SRC" ] || { echo "no source at $SRC — is backends/ mounted?" >&2; exit 1; }

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_SRC_DIR=/opt/triton-src \
  -DCMAKE_INSTALL_PREFIX="$BUILD/install"

cmake --build "$BUILD" -j"$(nproc)"

mkdir -p "$DEST"
cp "$BUILD/libtriton_tts_stream.so" "$DEST/"
echo "installed -> $DEST/libtriton_tts_stream.so"
echo "reload with: curl -X POST localhost:8000/v2/repository/models/tts_stream/load"
