#!/usr/bin/env bash
# Push the working tree to the GPU box over direct SSH.
#
#   source .box.env && bash scripts/sync.sh
#   bash scripts/sync.sh root@1.2.3.4 22106          # or pass explicitly
#
# Uses rsync when available (incremental, deletes removed files). Git Bash on Windows
# ships no rsync, so the fallback is tar-over-ssh, which needs nothing but OpenSSH and
# is plenty fast for a tree this size.
#
# Git remains the durable path; this is the tight iteration loop.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.box.env" ] && source "$REPO_ROOT/.box.env"

HOST="${1:-${BOX_HOST:?set BOX_HOST or pass user@host}}"
PORT="${2:-${BOX_PORT:-22}}"
KEY="${BOX_KEY:-$HOME/.ssh/id_ed25519}"
DIR="${REMOTE_DIR:-${BOX_DIR:-/workspace/streaming-tts-serving}}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i "$KEY" -p "$PORT")

EXCLUDES=(
  '.git' '__pycache__' '*.pyc' '.venv' 'venv'
  'models' 'results' '*.onnx' '*.plan' '*.engine'
  'backends/*/build' '.box.env'
)

cd "$REPO_ROOT"

if command -v rsync >/dev/null 2>&1; then
  args=()
  for e in "${EXCLUDES[@]}"; do args+=(--exclude "$e"); done
  rsync -az --delete --info=stats1 -e "ssh ${SSH_OPTS[*]}" "${args[@]}" ./ "$HOST:$DIR/"
else
  echo "rsync unavailable — using tar over ssh"
  args=()
  for e in "${EXCLUDES[@]}"; do args+=(--exclude="$e"); done
  # --owner/--group: Windows uids mean nothing on Linux and make tar noisy on extract.
  tar czf - --owner=0 --group=0 --numeric-owner "${args[@]}" . \
    | ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$DIR' && tar xzf - -C '$DIR' --no-same-owner && chmod +x '$DIR'/scripts/*.sh"
fi

ssh "${SSH_OPTS[@]}" "$HOST" "echo \"synced: \$(find '$DIR' -type f -not -path '*/.git/*' | wc -l) files at $DIR\""
