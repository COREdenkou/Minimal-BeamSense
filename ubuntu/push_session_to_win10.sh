#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export SMB_HOST=<WIN10_IP>
#   export SMB_SHARE=<SMB_SHARE_NAME>
#   export SMB_USER=<SMB_USERNAME>
#   export SMB_PASS='<SMB_PASSWORD>'
#   ./push_session_to_win10.sh <session_name> [source_root] [remote_base_subdir] [config_json]
#
# Also supports:
#   export SMB_AUTHFILE=/path/to/authfile
#
# Default remote target:
#   //<WIN10_IP>/<SMB_SHARE_NAME>/<REMOTE_BASE_SUBDIR>/<session_name>/

SESSION_NAME="${1:?usage: push_session_to_win10.sh <session_name> [source_root] [remote_base_subdir] [config_json]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON="${4:-$SCRIPT_DIR/config/peers.json}"

if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "[ERROR] config file not found: $CONFIG_JSON" >&2
  exit 1
fi

DEFAULT_CAPTURE_ROOT="$(python3 - <<'PY' "$CONFIG_JSON"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg.get("default_capture_root", ""))
PY
)"

DEFAULT_REMOTE_BASE_SUBDIR="$(python3 - <<'PY' "$CONFIG_JSON"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg.get("win10_share_base_subdir", "MinimalBeamSense/inbox"))
PY
)"

SOURCE_ROOT="${2:-$DEFAULT_CAPTURE_ROOT}"
REMOTE_BASE_SUBDIR="${3:-$DEFAULT_REMOTE_BASE_SUBDIR}"

SMB_HOST="${SMB_HOST:-}"
SMB_SHARE="${SMB_SHARE:-<SMB_SHARE_NAME>}"
SMB_USER="${SMB_USER:-}"
SMB_PASS="${SMB_PASS:-}"
SMB_AUTHFILE="${SMB_AUTHFILE:-}"

if [[ -z "$SMB_HOST" ]]; then
  echo "[ERROR] SMB_HOST is empty; set SMB_HOST=<WIN10_IP>" >&2
  exit 1
fi

if [[ -z "$SMB_USER" && -z "$SMB_AUTHFILE" ]]; then
  echo "[ERROR] set SMB_USER/SMB_PASS or SMB_AUTHFILE" >&2
  exit 1
fi

SESSION_DIR="$SOURCE_ROOT/$SESSION_NAME"
if [[ ! -d "$SESSION_DIR" ]]; then
  echo "[ERROR] session dir not found: $SESSION_DIR" >&2
  exit 1
fi

PCAP="$SESSION_DIR/capture.pcapng"
SUMMARY_TXT="$SESSION_DIR/summary.txt"
SUMMARY_JSON="$SESSION_DIR/summary.json"
MANIFEST_JSON="$SESSION_DIR/session_manifest.json"

if [[ ! -f "$PCAP" ]]; then
  echo "[ERROR] missing $PCAP" >&2
  exit 1
fi

CSV_FILES=("$SESSION_DIR"/capture_*.csv)

TMP_AUTH=""
if [[ -z "$SMB_AUTHFILE" ]]; then
  if [[ -z "$SMB_USER" || -z "$SMB_PASS" ]]; then
    echo "[ERROR] when SMB_AUTHFILE is not used, SMB_USER and SMB_PASS are required" >&2
    exit 1
  fi
  TMP_AUTH="$(mktemp)"
  chmod 600 "$TMP_AUTH"
  {
    echo "username = $SMB_USER"
    echo "password = $SMB_PASS"
  } > "$TMP_AUTH"
  SMB_AUTHFILE="$TMP_AUTH"
fi

TMP_CMD="$(mktemp)"
cleanup() {
  rm -f "$TMP_CMD"
  if [[ -n "$TMP_AUTH" ]]; then
    rm -f "$TMP_AUTH"
  fi
}
trap cleanup EXIT

REMOTE_SESSION_SUBDIR="$REMOTE_BASE_SUBDIR/$SESSION_NAME"

{
  echo "mkdir \"$REMOTE_SESSION_SUBDIR\""
  echo "cd \"$REMOTE_SESSION_SUBDIR\""

  echo "put \"$PCAP\" \"capture.pcapng\""

  if [[ -f "$SUMMARY_TXT" ]]; then
    echo "put \"$SUMMARY_TXT\" \"summary.txt\""
  fi

  if [[ -f "$SUMMARY_JSON" ]]; then
    echo "put \"$SUMMARY_JSON\" \"summary.json\""
  fi

  if [[ -f "$MANIFEST_JSON" ]]; then
    echo "put \"$MANIFEST_JSON\" \"session_manifest.json\""
  fi

  for f in "${CSV_FILES[@]}"; do
    if [[ -f "$f" ]]; then
      base="$(basename "$f")"
      echo "put \"$f\" \"$base\""
    fi
  done

  echo "ls"
} > "$TMP_CMD"

echo "[INFO] pushing session: $SESSION_NAME"
echo "[INFO] source: $SESSION_DIR"
echo "[INFO] remote: //$SMB_HOST/$SMB_SHARE/$REMOTE_SESSION_SUBDIR"

smbclient "//$SMB_HOST/$SMB_SHARE" -A "$SMB_AUTHFILE" -c "$(tr '\n' ';' < "$TMP_CMD")"

echo "[OK] push done"
