#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./qc_and_export_session.sh <session_name_or_session_dir> [ap_bssid] [config_json]
#
# Example:
#   ./qc_and_export_session.sh example_session_01
#
# Outputs inside the session directory:
#   - summary.txt
#   - summary.json
#   - cbf_by_peer.csv
#   - capture_<peer>.csv   for each configured peer
#
# Notes:
# - Peer MACs are loaded from config/peers.json
# - AP BSSID defaults to peers.json["ap_bssid"] if not provided explicitly

INPUT="${1:?usage: qc_and_export_session.sh <session_name_or_session_dir> [ap_bssid] [config_json]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON="${3:-$SCRIPT_DIR/config/peers.json}"

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

AP_DEFAULT="$(python3 - <<'PY' "$CONFIG_JSON"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg.get("ap_bssid", ""))
PY
)"

AP="${2:-$AP_DEFAULT}"

if [[ -z "$AP" ]]; then
  echo "[ERROR] AP BSSID is empty; pass it explicitly or set ap_bssid in peers.json" >&2
  exit 1
fi

if [[ -d "$INPUT" ]]; then
  SESSION_DIR="$INPUT"
else
  if [[ -z "$DEFAULT_CAPTURE_ROOT" ]]; then
    echo "[ERROR] default_capture_root is empty in peers.json, and INPUT is not a directory" >&2
    exit 1
  fi
  SESSION_DIR="$DEFAULT_CAPTURE_ROOT/$INPUT"
fi

PCAP="$SESSION_DIR/capture.pcapng"
if [[ ! -f "$PCAP" ]]; then
  echo "[ERROR] pcap not found: $PCAP" >&2
  exit 1
fi

SUMMARY_TXT="$SESSION_DIR/summary.txt"
SUMMARY_JSON="$SESSION_DIR/summary.json"
CBF_BY_PEER_CSV="$SESSION_DIR/cbf_by_peer.csv"

# Load peers as "name<TAB>mac<TAB>backend_name<TAB>backend_ip"
mapfile -t PEER_LINES < <(
python3 - <<'PY' "$CONFIG_JSON"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
for name, info in cfg.get("peers", {}).items():
    mac = info.get("mac", "")
    backend_name = info.get("backend_name", "")
    backend_ip = info.get("backend_ip", "")
    print(f"{name}\t{mac}\t{backend_name}\t{backend_ip}")
PY
)

if [[ "${#PEER_LINES[@]}" -eq 0 ]]; then
  echo "[ERROR] no peers found in $CONFIG_JSON" >&2
  exit 1
fi

# Robust AP filter: any of the common 802.11 address fields hits the AP
APF="(wlan.bssid==$AP || wlan.sa==$AP || wlan.da==$AP || wlan.ta==$AP || wlan.ra==$AP)"

NDPA=$(tshark -r "$PCAP" -Y "wlan.fc.type_subtype==0x0015 && $APF" -T fields -e frame.number | wc -l)
BFPOLL=$(tshark -r "$PCAP" -Y "wlan.fc.type_subtype==0x0014 && $APF" -T fields -e frame.number | wc -l)
CBF_TOTAL=$(tshark -r "$PCAP" -Y "wlan.vht.compressed_beamforming_report && $APF" -T fields -e frame.number | wc -l)

# Write text summary header
{
  echo "session_dir=$SESSION_DIR"
  echo "pcap=$PCAP"
  echo "ap=$AP"
  echo "config_json=$CONFIG_JSON"
  echo "ndpa=$NDPA"
  echo "bfpoll=$BFPOLL"
  echo "cbf_total=$CBF_TOTAL"
} | tee "$SUMMARY_TXT"

echo "[INFO] per-peer CBF counts:" | tee -a "$SUMMARY_TXT"

# CSV header
echo "peer_id,mac,backend_name,backend_ip,cbf_count" > "$CBF_BY_PEER_CSV"

# Collect per-peer info for JSON
TMP_JSON_LINES="$(mktemp)"
trap 'rm -f "$TMP_JSON_LINES"' EXIT
: > "$TMP_JSON_LINES"

for LINE in "${PEER_LINES[@]}"; do
  IFS=$'\t' read -r PEER_ID MAC BACKEND_NAME BACKEND_IP <<< "$LINE"

  if [[ -z "$MAC" ]]; then
    echo "[WARN] peer $PEER_ID has empty MAC, skipping" | tee -a "$SUMMARY_TXT"
    continue
  fi

  FILTER="wlan.vht.compressed_beamforming_report && $APF && (wlan.sa==$MAC || wlan.ta==$MAC || wlan.ra==$MAC || wlan.da==$MAC)"
  COUNT=$(tshark -r "$PCAP" -Y "$FILTER" -T fields -e frame.number | wc -l)

  echo "cbf_${PEER_ID}=$COUNT mac=$MAC backend=$BACKEND_NAME ip=$BACKEND_IP" | tee -a "$SUMMARY_TXT"
  echo "$PEER_ID,$MAC,$BACKEND_NAME,$BACKEND_IP,$COUNT" >> "$CBF_BY_PEER_CSV"

  CSV="$SESSION_DIR/capture_${PEER_ID}.csv"
  {
    echo "No,Time"
    tshark -r "$PCAP" \
      -Y "$FILTER" \
      -T fields -E separator=, \
      -e frame.number -e frame.time_relative
  } > "$CSV"

  echo "[INFO] exported $CSV" | tee -a "$SUMMARY_TXT"

  printf '%s\t%s\t%s\t%s\t%s\n' "$PEER_ID" "$MAC" "$BACKEND_NAME" "$BACKEND_IP" "$COUNT" >> "$TMP_JSON_LINES"
done

echo | tee -a "$SUMMARY_TXT"
echo "[INFO] raw peer-mac distribution (all addr fields except AP):" | tee -a "$SUMMARY_TXT"
RAW_DIST="$(
tshark -r "$PCAP" \
  -Y "wlan.vht.compressed_beamforming_report && $APF" \
  -T fields -e wlan.sa -e wlan.ta -e wlan.ra -e wlan.da \
| tr '\t' '\n' \
| sed '/^$/d' \
| grep -vi "^$AP$" \
| sort | uniq -c
)"
echo "$RAW_DIST" | tee -a "$SUMMARY_TXT"

# Write structured JSON summary
python3 - <<'PY' "$SUMMARY_JSON" "$SESSION_DIR" "$PCAP" "$AP" "$CONFIG_JSON" "$NDPA" "$BFPOLL" "$CBF_TOTAL" "$TMP_JSON_LINES" "$RAW_DIST"
import json, sys

summary_json, session_dir, pcap, ap, config_json, ndpa, bfpoll, cbf_total, peer_tsv, raw_dist = sys.argv[1:11]

per_peer = {}
with open(peer_tsv, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line:
            continue
        peer_id, mac, backend_name, backend_ip, count = line.split('\t')
        per_peer[peer_id] = {
            "mac": mac,
            "backend_name": backend_name,
            "backend_ip": backend_ip,
            "cbf_count": int(count),
        }

obj = {
    "session_dir": session_dir,
    "pcap": pcap,
    "ap_bssid": ap,
    "config_json": config_json,
    "ndpa": int(ndpa),
    "bfpoll": int(bfpoll),
    "cbf_total": int(cbf_total),
    "per_peer": per_peer,
    "raw_peer_mac_distribution_text": raw_dist,
}

with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)
PY

echo
echo "[INFO] summary saved to $SUMMARY_TXT"
echo "[INFO] summary saved to $SUMMARY_JSON"
echo "[INFO] per-peer csv saved to $CBF_BY_PEER_CSV"
