#!/usr/bin/env bash
# Provision the GPU box for streaming-tts-serving.
#
# Target environment: a RunPod pod started FROM the Triton NGC image
# (nvcr.io/nvidia/tritonserver:24.08-py3). The pod is itself a container, so there is
# no Docker available inside it — Triton, TensorRT and the CUDA stack come from the
# image, and the observability services install as plain static binaries.
#
# If you are instead on a real VM with Docker, use docker/docker-compose.yml and skip
# this script entirely. See docker/README.md.
#
# Idempotent: safe to re-run.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Recover the image's PATH / LD_LIBRARY_PATH / NVIDIA_* vars, which an sshd-spawned
# shell does not inherit. Without this, tritonserver and trtexec look absent.
source "$REPO_ROOT/scripts/container_env.sh"

# Storage split. /workspace is RunPod's persistent volume but in some datacenters it is a
# NETWORK filesystem (MooseFS) — fine for a few large files, poor for the tens of
# thousands of small ones in a venv, where it slows both install and every later import.
# So: big-and-expensive-to-refetch goes on the volume, many-small-files stays local even
# though that means re-running this script after a pod rebuild (a few minutes, scripted).
WORK="${WORK:-/workspace}"     # persistent: models, TRT engines, results, prometheus data
LOCAL="${LOCAL:-/opt/tts}"     # ephemeral, fast: venv, tool binaries, build trees
VENV="${VENV:-$LOCAL/venv}"
OPT="${OPT:-$LOCAL/tools}"

PROM_VERSION="${PROM_VERSION:-2.54.1}"
GRAFANA_VERSION="${GRAFANA_VERSION:-11.2.0}"
JAEGER_VERSION="${JAEGER_VERSION:-1.60.0}"
OTELCOL_VERSION="${OTELCOL_VERSION:-0.105.0}"
GO_VERSION="${GO_VERSION:-1.22.5}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mxx  %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ sanity
log "Environment"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — wrong box"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

command -v tritonserver >/dev/null \
  || die "tritonserver not found. This pod was not started from the Triton NGC image — see docs/GPU_BOX.md."

# `tritonserver --version` is not a flag; the image carries the versions in the env and
# in TRITON_VERSION. The container release (24.08) is what the backend SDK is tagged by,
# which is a different number from the server version (2.49.0).
TRITON_RELEASE="${NVIDIA_TRITON_SERVER_VERSION:-}"
[ -z "$TRITON_RELEASE" ] && TRITON_RELEASE="$(cat /opt/tritonserver/NVIDIA_TRITON_RELEASE 2>/dev/null || true)"
[ -z "$TRITON_RELEASE" ] && die "cannot determine Triton container release — set NVIDIA_TRITON_SERVER_VERSION"
echo "triton: server $(cat /opt/tritonserver/TRITON_VERSION 2>/dev/null) (container ${TRITON_RELEASE})"

# The NGC Triton image ships the TensorRT C++ libs, the tensorrt backend and trtexec,
# but not the python bindings. trtexec is enough to build engines; the bindings are only
# a convenience for scripted builds, so install them but do not fail without them.
python3 -c 'import tensorrt as t; print("tensorrt python:", t.__version__)' 2>/dev/null \
  || warn "TensorRT python bindings absent — engine builds will use trtexec ($(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec))"

mkdir -p "$WORK" "$LOCAL" "$OPT" || die "cannot create work dirs"
echo "persistent: $(df -h "$WORK" | tail -1)"
echo "local:      $(df -h / | tail -1)"

# ------------------------------------------------------------------ apt deps
log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential cmake git curl wget rsync jq unzip pkg-config tmux \
  python3-venv python3-dev \
  rapidjson-dev libb64-dev libssl-dev zlib1g-dev libarchive-dev \
  espeak-ng libsndfile1 sox ffmpeg \
  || die "apt install failed"

# ------------------------------------------------------------------ python
# A venv keeps torch and transformers out of Triton's own site-packages. Triton does not
# need the torch *python* package; only our export and baseline scripts do.
log "Python venv at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools

# TensorRT python bindings MUST match the container's TensorRT, or engines built here
# will not load in Triton's tensorrt backend. `pip install tensorrt` resolves to the
# newest release (11.x) against a 10.3 container — a mismatch that stays invisible until
# the model fails to load much later. Pin to whatever trtexec reports.
TRT_VER="$(trtexec --help 2>&1 | grep -oP 'TensorRT v\K[0-9]+' | head -1)"
if [ -n "$TRT_VER" ]; then
  # v100300 -> 10.3.0
  TRT_PIN="$((10#${TRT_VER:0:2})).$((10#${TRT_VER:2:2})).$((10#${TRT_VER:4:2}))"
  log "Pinning TensorRT python bindings to ${TRT_PIN} (matches trtexec)"
else
  TRT_PIN=""
  warn "could not detect TensorRT version from trtexec; skipping python binding pin"
fi

# numpy MUST match the container's, because services.sh puts this venv on PYTHONPATH so
# Triton's python backends can import torch and transformers — and PYTHONPATH always
# precedes site-packages in sys.path, so the venv's numpy wins whether or not it was
# "appended". Triton's python backend is compiled against numpy 1.x; handing it 2.x
# corrupts string tensor deserialization with an unpack_from error that points at
# triton_python_backend_utils and says nothing about numpy.
SYS_NUMPY="$(/usr/bin/python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "")"
if [ -n "$SYS_NUMPY" ]; then
  log "Pinning venv numpy to the container's ${SYS_NUMPY}"
fi

log "Installing torch (cu12x) — this is ~2.5 GB"
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124 \
  || warn "torch install failed; check the CUDA wheel index matches this image"
"$VENV/bin/pip" install -q -r "$REPO_ROOT/export/requirements.txt" || warn "export deps incomplete"
"$VENV/bin/pip" install -q fastapi "uvicorn[standard]" aiohttp || warn "baseline deps incomplete"
"$VENV/bin/pip" install -q "tritonclient[grpc]" || warn "tritonclient missing — streaming/client.py will not run"
# Last, so nothing else can pull numpy 2 back in as a transitive dependency.
if [ -n "$SYS_NUMPY" ]; then
  "$VENV/bin/pip" install -q "numpy==${SYS_NUMPY}" || warn "could not pin numpy to ${SYS_NUMPY}"
fi
[ -n "$TRT_PIN" ] && { "$VENV/bin/pip" install -q "tensorrt==${TRT_PIN}" \
  || warn "tensorrt==${TRT_PIN} unavailable; build engines with trtexec instead"; }

# Python backend deps go into the SYSTEM python, not the venv — Triton's python backend
# stub runs against the container's interpreter.
log "Installing tts_frontend deps into system python"
pip3 install -q --no-cache-dir phonemizer==3.2.1 inflect==7.3.1 unidecode==1.3.8 numpy \
  || warn "frontend deps incomplete"
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1

# ------------------------------------------------------------------ go
if ! command -v go >/dev/null 2>&1 && [ ! -x "$OPT/go/bin/go" ]; then
  log "Installing Go ${GO_VERSION}"
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tgz \
    && tar -C "$OPT" -xzf /tmp/go.tgz && rm -f /tmp/go.tgz
fi
export PATH="$OPT/go/bin:$PATH"
echo "go: $(go version 2>/dev/null || echo MISSING)"

# ------------------------------------------------- triton backend SDK sources
# Needed to compile the tts_stream C++ backend against this server's exact ABI.
TRITON_TAG="r${TRITON_RELEASE}"
log "Fetching Triton backend SDK (${TRITON_TAG})"
mkdir -p "$OPT/triton-src"
for repo in backend core common; do
  d="$OPT/triton-src/$repo"
  if [ -d "$d/.git" ]; then
    git -C "$d" fetch -q --depth 1 origin "$TRITON_TAG" 2>/dev/null && git -C "$d" checkout -q FETCH_HEAD || true
  else
    git clone -q --depth 1 -b "$TRITON_TAG" "https://github.com/triton-inference-server/${repo}.git" "$d" \
      || warn "could not clone $repo at $TRITON_TAG — falling back to main" \
      && [ -d "$d" ] || git clone -q --depth 1 "https://github.com/triton-inference-server/${repo}.git" "$d"
  fi
done

# ------------------------------------------------------------ observability
# These never needed Docker; compose was only ever packaging convenience.
fetch_tar() { # url, strip-dir-name, dest
  local url="$1" name="$2"
  [ -d "$OPT/$name" ] && { echo "  $name already installed"; return 0; }
  echo "  fetching $name"
  curl -fsSL "$url" -o /tmp/"$name".tgz || { warn "download failed: $name"; return 1; }
  mkdir -p "$OPT/$name"
  tar -xzf /tmp/"$name".tgz -C "$OPT/$name" --strip-components=1
  rm -f /tmp/"$name".tgz
}

log "Installing observability binaries into $OPT"
fetch_tar "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz" prometheus
fetch_tar "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz" grafana
fetch_tar "https://github.com/jaegertracing/jaeger/releases/download/v${JAEGER_VERSION}/jaeger-${JAEGER_VERSION}-linux-amd64.tar.gz" jaeger
fetch_tar "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}/otelcol-contrib_${OTELCOL_VERSION}_linux_amd64.tar.gz" otelcol

# No DCGM exporter here: it wants privileged access we do not have inside a pod, and
# Triton already exports nv_gpu_utilization / nv_gpu_memory_used_bytes / nv_gpu_power_usage
# on its own metrics endpoint, which covers what the dashboards need.

# ------------------------------------------------------------------ shell env
ENVFILE="$WORK/env.sh"
cat > "$ENVFILE" <<EOF
export WORK="$WORK"
export LOCAL="$LOCAL"
export OPT="$OPT"
export VENV="$VENV"
export REPO="$REPO_ROOT"
export PATH="$OPT/go/bin:/usr/src/tensorrt/bin:\$PATH"
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
export TRITON_SRC_DIR="$OPT/triton-src"
export TRITON_RELEASE="$TRITON_RELEASE"
export TTS_MODEL_DIR="$WORK/models"
alias venv="source $VENV/bin/activate"
EOF
grep -q "source $ENVFILE" ~/.bashrc 2>/dev/null || echo "source $ENVFILE" >> ~/.bashrc

log "Done."
cat <<EOF

  source $ENVFILE && source $VENV/bin/activate

  Next:
    python export/fetch_model.py        # download VITS into \$TTS_MODEL_DIR
    python export/inspect_model.py      # real shapes + per-module timings  (M2)
    bash scripts/services.sh start      # prometheus, grafana, jaeger, otel

  Reachable via RunPod's HTTP proxy (no SSH tunnel needed):
    grafana   https://<POD_ID>-3000.proxy.runpod.net
    prometheus https://<POD_ID>-9090.proxy.runpod.net
    jaeger    https://<POD_ID>-16686.proxy.runpod.net
    triton    https://<POD_ID>-8000.proxy.runpod.net/v2/health/ready

  Terminate the pod when you stop working — provisioning is scripted, rebuilding is cheap.

EOF
