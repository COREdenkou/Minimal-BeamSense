#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./capture_session.sh <session_name> [duration_sec] [if_mon] [out_root]
#
# Example:
#   ./capture_session.sh example_session_01 120 <MONITOR_IFACE> <CAPTURE_ROOT>

SESSION_NAME="${1:?usage: capture_session.sh <session_name> [duration_sec] [if_mon] [out_root]}"
DURATION="${2:-60}"
IF_MON="${3:-<MONITOR_IFACE>}"
OUT_ROOT="${4:-${CAPTURE_ROOT:-$HOME/captures}}"

OUTDIR="$OUT_ROOT/$SESSION_NAME"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

META_FILE="$OUTDIR/capture_meta.txt"

{
  echo "session_name=$SESSION_NAME"
  echo "created_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "hostname=$(hostname)"
  echo "duration_sec=$DURATION"
  echo "monitor_interface=$IF_MON"
  echo "out_root=$OUT_ROOT"
  echo "outdir=$OUTDIR"
} > "$META_FILE"

echo "[INFO] session=$SESSION_NAME"
echo "[INFO] outdir=$OUTDIR"
echo "[INFO] IF_MON=$IF_MON duration=$DURATION"
echo "[INFO] meta=$META_FILE"

status=0
sudo timeout -s INT -k 3 "$DURATION" tcpdump -i "$IF_MON" -s 0 -U -w capture.pcapng || status=$?

# timeout exiting at the requested duration is expected and should not be treated as failure.
if [[ "$status" -ne 0 && "$status" -ne 124 && "$status" -ne 130 && "$status" -ne 137 && "$status" -ne 143 ]]; then
  echo "[ERROR] tcpdump failed with status=$status" >&2
  exit "$status"
fi

echo "[INFO] capture done: $OUTDIR/capture.pcapng"
ls -lh capture.pcapng
