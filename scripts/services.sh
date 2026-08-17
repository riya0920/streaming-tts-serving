#!/usr/bin/env bash
# Start/stop the observability services natively (no Docker — the pod is a container).
#
#   bash scripts/services.sh start|stop|status|logs <name>
#
# Port map (all also reachable through RunPod's HTTPS proxy at <POD_ID>-<port>...):
#   3000   grafana
#   9090   prometheus
#   16686  jaeger UI
#   4317/4318  otel collector  (OTLP in — what Triton and the gateway point at)
#   5317/5318  jaeger OTLP     (moved off 4317 so it does not collide with the collector)
#   8000/8001/8002  triton http/grpc/metrics
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# tritonserver is only on PATH once the image's env is recovered — see container_env.sh.
source "$REPO/scripts/container_env.sh"

WORK="${WORK:-/workspace}"
LOCAL="${LOCAL:-/opt/tts}"
OPT="${OPT:-$LOCAL/tools}"
RUN="$WORK/run"
LOGS="$WORK/logs"
DATA="$WORK/data"
mkdir -p "$RUN" "$LOGS" "$DATA"

log() { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
err() { printf '\033[1;31mxx  %s\033[0m\n' "$*" >&2; }

_pidfile() { echo "$RUN/$1.pid"; }

_running() {
  local pf; pf="$(_pidfile "$1")"
  [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null
}

_spawn() { # name, command...
  local name="$1"; shift
  if _running "$name"; then echo "  $name already running (pid $(cat "$(_pidfile "$name")"))"; return 0; fi
  nohup "$@" >"$LOGS/$name.log" 2>&1 &
  echo $! > "$(_pidfile "$name")"
  sleep 1
  if _running "$name"; then
    echo "  $name started (pid $(cat "$(_pidfile "$name")"))"
  else
    err "$name failed to start — tail $LOGS/$name.log"
    tail -20 "$LOGS/$name.log" >&2
    return 1
  fi
}

_kill() {
  local name="$1" pf; pf="$(_pidfile "$name")"
  if _running "$name"; then
    kill "$(cat "$pf")" 2>/dev/null
    for _ in $(seq 20); do _running "$name" || break; sleep 0.25; done
    _running "$name" && kill -9 "$(cat "$pf")" 2>/dev/null
    echo "  $name stopped"
  else
    echo "  $name not running"
  fi
  rm -f "$pf"
}

start_otelcol() {
  _spawn otelcol "$OPT/otelcol/otelcol-contrib" --config="$REPO/observability/otel-collector.yaml"
}

start_jaeger() {
  # OTLP receivers moved off the default 4317/4318 to leave those to the collector.
  # No comments between a \ continuation and the next line: bash folds the
  # comment into the command and silently eats the rest of it.
  # (--query.host-port was removed in Jaeger 1.6x; the UI defaults to 16686.)
  COLLECTOR_OTLP_ENABLED=true \
  SPAN_STORAGE_TYPE=memory \
  _spawn jaeger "$OPT/jaeger/jaeger-all-in-one" \
    --collector.otlp.grpc.host-port=:5317 \
    --collector.otlp.http.host-port=:5318
}

start_prometheus() {
  _spawn prometheus "$OPT/prometheus/prometheus" \
    --config.file="$REPO/observability/prometheus.yml" \
    --storage.tsdb.path="$DATA/prometheus" \
    --storage.tsdb.retention.time=15d \
    --web.listen-address=:9090 \
    --web.enable-lifecycle
}

start_grafana() {
  GF_PATHS_PROVISIONING="$REPO/observability/grafana/provisioning" \
  GF_PATHS_DATA="$DATA/grafana" \
  GF_PATHS_LOGS="$LOGS/grafana" \
  GF_SERVER_HTTP_PORT=3000 \
  GF_AUTH_ANONYMOUS_ENABLED=true \
  GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
  GF_AUTH_DISABLE_LOGIN_FORM=true \
  _spawn grafana "$OPT/grafana/bin/grafana" server --homepath "$OPT/grafana"
}

# One Triton per GPU, each pinned with CUDA_VISIBLE_DEVICES and offset by 10 ports.
#
# A single RTX 6000 Ada holds ~1,067 sessions at 10% duty (p99 149ms) and saturates at
# aggregate RTF ~205 — a hardware ceiling. More capacity is horizontal, so the gateway
# takes a comma-separated endpoint list and least-in-flight routes across them.
#
# Separate processes rather than one Triton with multiple GPU instance_groups: this way a
# wedged backend takes down one GPU's worth of capacity instead of all of it, and each
# server's metrics are attributable to a specific card.
start_triton_all() {
  local n
  n="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
  [ "${n:-0}" -lt 1 ] && n=1
  echo "  launching $n triton instance(s), one per GPU"
  for i in $(seq 0 $((n - 1))); do
    start_triton "$i"
  done
  echo
  echo "  point the gateway at:"
  local addrs=""
  for i in $(seq 0 $((n - 1))); do
    addrs="${addrs}${addrs:+,}localhost:$((8001 + i * 10))"
  done
  echo "    TRITON_GRPC_ADDR=$addrs"
}

start_triton() {
  local gpu="${1:-0}"
  local http=$((8000 + gpu * 10))
  local grpc=$((8001 + gpu * 10))
  local met=$((8002 + gpu * 10))
  local name="triton${gpu}"
  local repo_arg="$REPO/model_repo"
  if [ -z "$(ls -A "$repo_arg" 2>/dev/null)" ]; then
    err "model_repo is empty — nothing to serve yet (expected until M5)"
    return 1
  fi

  # Triton's python backend stubs inherit the server's environment, so pointing
  # PYTHONPATH at the venv lets tts_frontend and vits_frontend import torch and
  # transformers without a second multi-gigabyte install into the container's python.
  # Appended, not prepended: Triton's own packages must keep priority.
  local vsp
  vsp="$(ls -d "${VENV:-/opt/tts/venv}"/lib/python*/site-packages 2>/dev/null | head -1)"
  if [ -n "$vsp" ]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$vsp"
    echo "  python backends will see $vsp"
  else
    err "venv site-packages not found — python backends will fail to import torch"
  fi
  export TTS_MODEL_DIR="${WORK}/models"
  # Pin this server to one card. Without it every instance grabs GPU 0 and the
  # "one Triton per GPU" split is cosmetic.
  export CUDA_VISIBLE_DEVICES="$gpu"

  _spawn "$name" tritonserver \
    --model-repository="$repo_arg" \
    --http-port="$http" --grpc-port="$grpc" --metrics-port="$met" \
    --allow-metrics=true \
    --trace-config=mode=opentelemetry \
    --trace-config=opentelemetry,url=http://localhost:4318/v1/traces \
    --trace-config=rate=100 \
    --log-verbose=0
  echo "    gpu $gpu -> http $http / grpc $grpc / metrics $met"
}

case "${1:-status}" in
  start)
    shift
    # Triton is not in the default set: it has nothing to serve until M5 populates
    # model_repo/. Start it explicitly with `services.sh start triton`.
    if [ $# -eq 0 ]; then targets=(otelcol jaeger prometheus grafana); else targets=("$@"); fi
    # "triton" means every GPU; a bare start_triton would only ever start GPU 0.
    for i in "${!targets[@]}"; do
      [ "${targets[$i]}" = "triton" ] && targets[$i]="triton_all"
    done
    log "starting: ${targets[*]}"
    for t in "${targets[@]}"; do "start_$t"; done
    echo
    log "URLs (replace POD_ID with your pod id):"
    echo "  grafana     https://POD_ID-3000.proxy.runpod.net"
    echo "  prometheus  https://POD_ID-9090.proxy.runpod.net"
    echo "  jaeger      https://POD_ID-16686.proxy.runpod.net"
    ;;
  stop)
    shift
    targets=("$@")
    if [ $# -eq 0 ]; then
      targets=(grafana prometheus jaeger otelcol)
      for i in $(seq 0 7); do [ -f "$RUN/triton${i}.pid" ] && targets=("triton${i}" "${targets[@]}"); done
    fi
    log "stopping: ${targets[*]}"
    for t in "${targets[@]}"; do _kill "$t"; done
    ;;
  status)
    names=(otelcol jaeger prometheus grafana)
    for i in $(seq 0 7); do [ -f "$RUN/triton${i}.pid" ] && names=("triton${i}" "${names[@]}"); done
    for n in "${names[@]}"; do
      if _running "$n"; then printf "  %-11s UP   pid %s\n" "$n" "$(cat "$(_pidfile "$n")")"
      else printf "  %-11s down\n" "$n"; fi
    done
    ;;
  logs)
    tail -f "$LOGS/${2:?which service}.log"
    ;;
  *)
    echo "usage: services.sh start|stop|status|logs <name>" >&2; exit 2 ;;
esac
