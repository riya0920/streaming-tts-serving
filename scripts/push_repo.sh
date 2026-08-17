#!/usr/bin/env bash
# Push the working tree to the pod over RunPod's SSH proxy.
#
# The proxy gives an interactive shell and nothing else — no scp, no rsync, no port
# forwarding. So the tree is tarred, base64'd, and written back in pieces.
#
# Two limits bite here, and both cause SILENT truncation:
#   1. A pty in canonical mode truncates input lines at ~4 KB  -> chunk lines at 1800.
#   2. The tty input queue is small, and we transmit faster than the remote shell
#      executes, so a large payload overflows it mid-transfer -> split across several
#      SSH sessions, each small enough to drain.
# The transfer is verified by md5 at the end rather than trusted.
#
# If the pod exposes TCP 22, get the direct `ssh root@<ip> -p <port>` endpoint from the
# RunPod console and use scripts/sync.sh (rsync) instead — much faster, incremental.
#
#   bash scripts/push_repo.sh [remote_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${1:-/workspace/streaming-tts-serving}"
LINE=1800          # chars per printf line
BYTES_PER_SESSION=${BYTES_PER_SESSION:-14000}
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$REPO_ROOT"
# --owner/--group: the tree is authored on Windows, whose uids are meaningless on Linux
# and make tar noisy on extract.
tar czf "$TMP/repo.tgz" \
  --owner=0 --group=0 --numeric-owner \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='models' --exclude='results' \
  --exclude='*.onnx' --exclude='*.plan' \
  .

LOCAL_MD5="$(md5sum "$TMP/repo.tgz" | awk '{print $1}')"
base64 -w0 "$TMP/repo.tgz" > "$TMP/repo.b64"
B64=$(wc -c < "$TMP/repo.b64")
echo "tree $(wc -c < "$TMP/repo.tgz") bytes  md5 $LOCAL_MD5  ->  $B64 b64"

split -b "$BYTES_PER_SESSION" -d -a 3 "$TMP/repo.b64" "$TMP/part."
parts=("$TMP"/part.*)
echo "sending ${#parts[@]} sessions"

first=1
for p in "${parts[@]}"; do
  {
    if [ $first -eq 1 ]; then echo "rm -f /tmp/repo.b64"; fi
    fold -w "$LINE" "$p" | while IFS= read -r line; do
      # base64's alphabet contains no single quotes, so this quoting is safe.
      printf "printf '%%s' '%s' >> /tmp/repo.b64\n" "$line"
    done
    echo "wc -c < /tmp/repo.b64"
  } > "$TMP/payload.sh"
  first=0
  got=$(bash "$REPO_ROOT/scripts/rpod.sh" -f "$TMP/payload.sh" 2>/dev/null | tr -cd '0-9\n' | tail -1)
  printf "  %s -> remote has %s bytes\n" "$(basename "$p")" "${got:-?}"
done

echo "unpacking"
bash "$REPO_ROOT/scripts/rpod.sh" "
set -e
mkdir -p '$REMOTE_DIR'
base64 -d /tmp/repo.b64 > /tmp/repo.tgz
echo \"remote md5: \$(md5sum /tmp/repo.tgz | awk '{print \$1}')\"
echo \"expected  : $LOCAL_MD5\"
if [ \"\$(md5sum /tmp/repo.tgz | awk '{print \$1}')\" != '$LOCAL_MD5' ]; then
  echo 'CHECKSUM MISMATCH — transfer corrupted'; exit 1
fi
tar xzf /tmp/repo.tgz -C '$REMOTE_DIR' --no-same-owner
rm -f /tmp/repo.b64 /tmp/repo.tgz
chmod +x '$REMOTE_DIR'/scripts/*.sh
echo \"ok: \$(find '$REMOTE_DIR' -type f | wc -l) files at $REMOTE_DIR\"
"
