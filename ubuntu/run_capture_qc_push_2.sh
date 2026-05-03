#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_capture_qc_push_2.sh <session_name> [layout] [traffic_pattern] [active_peers] [preheat_sec] [capture_sec] [orientation_note] [notes]
#
# Example:
#   ./run_capture_qc_push_2.sh \
#     example_session_01 \
#     example_layout \
#     "iperf A=<RATE_A> B=<RATE_B>" \
#     rtax52_a,rtax52_b,rtax52_c \
#     10 \
#     120 \
#     "example orientation" \
#     "schedule: empty 0-30, one_person 40-70, two_person 80-110; IPs win10=<WIN10_IP> win11=<WIN11_IP> ubuntu=<UBUNTU_IP>; iperf A=<RATE_A> B=<RATE_B>"
#
# Notes:
# - This script orchestrates the Ubuntu side only.
# - It does NOT start iperf traffic for you; start traffic separately first.
# - Time schedule for physical actions should be aligned to the moment this script prints:
#       [INFO] CAPTURE START NOW
# - The session manifest is written twice:
#     before capture (to record intended setup)
#     after QC (to attach qc_summary if summary.json exists)

SESSION_NAME="${1:?usage: run_capture_qc_push_2.sh <session_name> [layout] [traffic_pattern] [active_peers] [preheat_sec] [capture_sec] [orientation_note] [notes]}"

LAYOUT="${2:-unspecified}"
TRAFFIC_PATTERN="${3:-unspecified}"
ACTIVE_PEERS="${4:-rtax52_a,rtax52_b,rtax52_c}"
PREHEAT_SEC="${5:-10}"
CAPTURE_SEC="${6:-60}"
ORIENTATION_NOTE="${7:-}"
NOTES="${8:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON="${SCRIPT_DIR}/config/peers.json"

CAPTURE_SCRIPT="$SCRIPT_DIR/capture_session.sh"
QC_SCRIPT="$SCRIPT_DIR/qc_and_export_session.sh"
MANIFEST_SCRIPT="$SCRIPT_DIR/write_session_manifest.py"
PUSH_SCRIPT="$SCRIPT_DIR/push_session_to_win10.sh"

for f in "$CAPTURE_SCRIPT" "$QC_SCRIPT" "$MANIFEST_SCRIPT" "$PUSH_SCRIPT" "$CONFIG_JSON"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f" >&2
    exit 1
  fi
done

read_config_value() {
  local key="$1"
  python3 - <<'PY' "$CONFIG_JSON" "$key"
import json, sys
cfg_path, key = sys.argv[1], sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get(key, ""))
PY
}

CAPTURE_ROOT="$(read_config_value default_capture_root)"
MONITOR_IF="$(read_config_value monitor_interface)"
AP_BSSID="$(read_config_value ap_bssid)"

if [[ -z "$CAPTURE_ROOT" ]]; then
  echo "[ERROR] default_capture_root is empty in $CONFIG_JSON" >&2
  exit 1
fi

if [[ -z "$MONITOR_IF" ]]; then
  echo "[ERROR] monitor_interface is empty in $CONFIG_JSON" >&2
  exit 1
fi

if [[ -z "$AP_BSSID" ]]; then
  echo "[ERROR] ap_bssid is empty in $CONFIG_JSON" >&2
  exit 1
fi

echo "[INFO] ==============================================="
echo "[INFO] Session Name    : $SESSION_NAME"
echo "[INFO] Layout          : $LAYOUT"
echo "[INFO] Traffic Pattern : $TRAFFIC_PATTERN"
echo "[INFO] Active Peers    : $ACTIVE_PEERS"
echo "[INFO] Preheat Sec     : $PREHEAT_SEC"
echo "[INFO] Capture Sec     : $CAPTURE_SEC"
echo "[INFO] Orientation     : $ORIENTATION_NOTE"
echo "[INFO] Notes           : $NOTES"
echo "[INFO] Config JSON     : $CONFIG_JSON"
echo "[INFO] Capture Root    : $CAPTURE_ROOT"
echo "[INFO] Monitor IF      : $MONITOR_IF"
echo "[INFO] AP BSSID        : $AP_BSSID"
echo "[INFO] Started At UTC  : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[INFO] ==============================================="

echo "[STEP] write pre-capture manifest"
"$MANIFEST_SCRIPT" "$SESSION_NAME" \
  --config "$CONFIG_JSON" \
  --capture-root "$CAPTURE_ROOT" \
  --layout "$LAYOUT" \
  --traffic-pattern "$TRAFFIC_PATTERN" \
  --active-peers "$ACTIVE_PEERS" \
  --preheat-sec "$PREHEAT_SEC" \
  --capture-sec "$CAPTURE_SEC" \
  --orientation-note "$ORIENTATION_NOTE" \
  --notes "$NOTES"

if [[ "$PREHEAT_SEC" =~ ^[0-9]+$ ]] && [[ "$PREHEAT_SEC" -gt 0 ]]; then
  echo "[STEP] preheat before capture"
  echo "[INFO] Please ensure foreground traffic is already running."
  echo "[INFO] Physical action schedule should start when this script prints:"
  echo "[INFO] CAPTURE START NOW"

  remaining="$PREHEAT_SEC"
  while [[ "$remaining" -gt 0 ]]; do
    echo "[INFO] capture starts in ${remaining}s"
    sleep 1
    remaining=$((remaining - 1))
  done
else
  echo "[STEP] preheat skipped (PREHEAT_SEC=$PREHEAT_SEC)"
fi

echo "[INFO] ==============================================="
echo "[INFO] CAPTURE START NOW"
echo "[INFO] Capture Window  : ${CAPTURE_SEC}s"
echo "[INFO] Capture UTC     : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[INFO] ==============================================="

echo "[STEP] capture session"
"$CAPTURE_SCRIPT" "$SESSION_NAME" "$CAPTURE_SEC" "$MONITOR_IF" "$CAPTURE_ROOT"

echo "[STEP] qc and export"
"$QC_SCRIPT" "$SESSION_NAME" "$AP_BSSID" "$CONFIG_JSON"

echo "[STEP] refresh manifest with qc summary"
"$MANIFEST_SCRIPT" "$SESSION_NAME" \
  --config "$CONFIG_JSON" \
  --capture-root "$CAPTURE_ROOT" \
  --layout "$LAYOUT" \
  --traffic-pattern "$TRAFFIC_PATTERN" \
  --active-peers "$ACTIVE_PEERS" \
  --preheat-sec "$PREHEAT_SEC" \
  --capture-sec "$CAPTURE_SEC" \
  --orientation-note "$ORIENTATION_NOTE" \
  --notes "$NOTES"

echo "[STEP] push to Win10"
"$PUSH_SCRIPT" "$SESSION_NAME" "$CAPTURE_ROOT" "" "$CONFIG_JSON"

echo "[OK] run_capture_qc_push_2 completed: $SESSION_NAME"
echo "[INFO] Finished At UTC : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
