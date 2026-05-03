#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="Write a session_manifest.json for the current Minimal BeamSense hardware session."
    )
    p.add_argument("session_name", help="Session name, e.g. example_session_01")
    p.add_argument(
        "--config",
        default="config/peers.json",
        help="Path to peers.json",
    )
    p.add_argument(
        "--capture-root",
        default=None,
        help="Override capture root; defaults to peers.json['default_capture_root']",
    )
    p.add_argument(
        "--layout",
        default="unspecified",
        help="Layout label, e.g. example_layout / bmain_layout",
    )
    p.add_argument(
        "--traffic-pattern",
        default="unspecified",
        help="Traffic pattern label, e.g. iperf A=<RATE_A> B=<RATE_B>",
    )
    p.add_argument(
        "--active-peers",
        default=None,
        help="Comma-separated peer ids, e.g. rtax52_a,rtax52_b,rtax52_c. Defaults to all peers in config.",
    )
    p.add_argument(
        "--preheat-sec",
        type=int,
        default=10,
        help="Preheat seconds used before starting capture",
    )
    p.add_argument(
        "--capture-sec",
        type=int,
        default=60,
        help="Capture duration in seconds",
    )
    p.add_argument(
        "--orientation-note",
        default="",
        help="Short note for antenna/orientation configuration",
    )
    p.add_argument(
        "--notes",
        default="",
        help="Any extra free-text note",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_json(args.config)
    capture_root = args.capture_root or cfg.get("default_capture_root")
    if not capture_root:
        raise RuntimeError("capture_root is empty; pass --capture-root or set default_capture_root in peers.json")

    session_dir = os.path.join(capture_root, args.session_name)
    os.makedirs(session_dir, exist_ok=True)

    peers_cfg = cfg.get("peers", {})
    if not peers_cfg:
        raise RuntimeError("No peers found in config")

    if args.active_peers:
        active_peers = [x.strip() for x in args.active_peers.split(",") if x.strip()]
    else:
        active_peers = list(peers_cfg.keys())

    for peer in active_peers:
        if peer not in peers_cfg:
            raise RuntimeError(f"Unknown active peer '{peer}' not found in peers.json")

    manifest = {
        "version": "beamsense_v2",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_name": args.session_name,
        "session_dir": session_dir,
        "ap_bssid": cfg.get("ap_bssid", ""),
        "monitor_interface": cfg.get("monitor_interface", ""),
        "capture": {
            "capture_root": capture_root,
            "preheat_sec": args.preheat_sec,
            "capture_sec": args.capture_sec,
        },
        "layout": args.layout,
        "traffic_pattern": args.traffic_pattern,
        "active_peers": active_peers,
        "orientation_note": args.orientation_note,
        "notes": args.notes,
        "peer_mapping": {},
    }

    for peer_id, info in peers_cfg.items():
        manifest["peer_mapping"][peer_id] = {
            "mac": info.get("mac", ""),
            "backend_name": info.get("backend_name", ""),
            "backend_ip": info.get("backend_ip", ""),
            "active_in_this_session": peer_id in active_peers,
            "notes": info.get("notes", ""),
        }

    # Attach QC summary if already available.
    summary_json_path = os.path.join(session_dir, "summary.json")
    if os.path.isfile(summary_json_path):
        try:
            manifest["qc_summary"] = load_json(summary_json_path)
        except Exception as e:
            manifest["qc_summary_error"] = str(e)

    out_path = os.path.join(session_dir, "session_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[INFO] manifest saved:", out_path)


if __name__ == "__main__":
    main()
