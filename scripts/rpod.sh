#!/usr/bin/env bash
# Run a script on the RunPod box and print only its output.
#
# RunPod's ssh.runpod.io proxy gives an interactive shell but ignores remote commands,
# so we feed the script over stdin. That means the terminal echoes everything back and
# wraps it in ANSI/bracketed-paste noise. This wrapper fences the real output between
# markers and strips the rest.
#
#   bash scripts/rpod.sh 'nvidia-smi -L'
#   bash scripts/rpod.sh -f scripts/provision.sh
#   echo 'some script' | bash scripts/rpod.sh -
#
# Env: RPOD_HOST (required, pods are ephemeral), RPOD_KEY, RPOD_TIMEOUT
set -uo pipefail

if [ -z "${RPOD_HOST:-}" ]; then
  echo "RPOD_HOST is not set. Example: RPOD_HOST=abc123-6441@ssh.runpod.io" >&2
  exit 2
fi
RPOD_HOST="${RPOD_HOST}"
RPOD_KEY="${RPOD_KEY:-$HOME/.ssh/id_ed25519}"
RPOD_TIMEOUT="${RPOD_TIMEOUT:-600}"

BEG="__RPOD_BEGIN_9f3a__"
END="__RPOD_END_9f3a__"

case "${1:-}" in
  -f) BODY="$(cat "$2")" ;;
  -)  BODY="$(cat)" ;;
  "") echo "usage: rpod.sh <command> | -f <file> | -" >&2; exit 2 ;;
  *)  BODY="$*" ;;
esac

# `set +e` inside so a failing line does not kill the shell before we print the marker;
# we capture and report the body's exit status explicitly.
payload=$(cat <<EOF
echo ${BEG}
set +e
$BODY
__rc=\$?
echo "${END}:\${__rc}"
exit
EOF
)

raw=$(printf '%s\n' "$payload" \
  | timeout "$RPOD_TIMEOUT" ssh -tt \
      -o StrictHostKeyChecking=no \
      -o ServerAliveInterval=20 \
      -o ConnectTimeout=20 \
      -i "$RPOD_KEY" "$RPOD_HOST" 2>&1)

clean=$(printf '%s' "$raw" \
  | tr -d '\r' \
  | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' \
  | sed 's/\x1b\][0-9];[^\x07]*\x07//g')

# The marker appears twice: once in the echoed input, once in real output. Take the
# text after the LAST BEGIN marker.
out=$(printf '%s\n' "$clean" | awk -v b="$BEG" -v e="$END" '
  # Reset done as well as buf: the marker pair appears once in the echoed input and
  # again in the real output, and a sticky done would suppress the second (real) block.
  $0 ~ b { buf=""; cap=1; fin=0; next }
  cap && !fin && $0 ~ e { fin=1; split($0, a, ":"); rc=a[2]; next }
  cap && !fin { buf = buf $0 "\n" }
  END { printf "%s", buf; exit (rc+0) }
')
rc=$?

printf '%s' "$out"
exit $rc
