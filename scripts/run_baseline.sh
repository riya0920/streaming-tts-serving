#!/usr/bin/env bash
# M1 — stand up the naive baseline and load it to failure, in both modes.
#
#   bash scripts/run_baseline.sh [duration_per_level]
#
# Both modes are measured because a strawman baseline is worse than no baseline:
#   async_blocking  the classic mistake (blocking torch call inside `async def`)
#   threadpool      the competent-naive version — this is the honest comparison basis
#
# Writes results/baseline_<mode>.json. Every later "N% faster" is measured against the
# threadpool file or it does not get claimed.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/container_env.sh"
[ -f /workspace/env.sh ] && source /workspace/env.sh
source "${VENV:-/opt/tts/venv}/bin/activate"

DURATION="${1:-20}"
PORT=8500
LEVELS="1,2,4,8,16,32,64,128"

cd "$REPO"
mkdir -p results logs

for MODE in threadpool async_blocking; do
  echo "=============================================================="
  echo "  baseline mode: $MODE"
  echo "=============================================================="

  BASELINE_MODE="$MODE" PORT="$PORT" TTS_MODEL_ID="${TTS_MODEL_ID:-facebook/mms-tts-eng}" \
    nohup python baseline/server.py > "logs/baseline_$MODE.log" 2>&1 &
  SRV=$!

  # Model load plus CUDA warmup takes a while; poll rather than guess.
  for i in $(seq 1 90); do
    if curl -fsS "http://localhost:$PORT/healthz" >/dev/null 2>&1; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then
      echo "server died on startup:"; tail -20 "logs/baseline_$MODE.log"; exit 1
    fi
    sleep 2
  done
  curl -fsS "http://localhost:$PORT/healthz" || { echo "never became healthy"; exit 1; }
  echo

  python baseline/loadtest.py --url "http://localhost:$PORT" \
    --levels "$LEVELS" --duration "$DURATION"

  kill $SRV 2>/dev/null
  wait $SRV 2>/dev/null
  sleep 3
done

echo
echo "results:"
ls -la results/
