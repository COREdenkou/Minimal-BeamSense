#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_get(d: Dict[str, Any], key: str, default=0):
    v = d.get(key, default)
    return default if v is None else v


def main():
    ap = argparse.ArgumentParser(description="Build one-page quality report for a session.")
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--fusion-json", required=True)
    ap.add_argument("--peer-a-json", required=True)
    ap.add_argument("--peer-b-json", required=True)
    ap.add_argument("--peer-c-json", required=True)
    ap.add_argument("--session-name", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    summary = load_json(Path(args.summary_json))
    fusion = load_json(Path(args.fusion_json))
    peer_a = load_json(Path(args.peer_a_json))
    peer_b = load_json(Path(args.peer_b_json))
    peer_c = load_json(Path(args.peer_c_json))

    per_peer_summary = summary.get("per_peer", {})
    per_peer_nonzero = fusion.get("per_peer_nonzero_windows", {})
    per_peer_mean_valid = fusion.get("mean_valid_frames_per_peer", {})

    peers = [
        {
            "peer_id": "rtax52_a",
            "backend_name": per_peer_summary.get("rtax52_a", {}).get("backend_name", "unknown"),
            "cbf_count": safe_get(per_peer_summary.get("rtax52_a", {}), "cbf_count", 0),
            "nonzero_windows": safe_get(per_peer_nonzero, "rtax52_a", 0),
            "mean_valid_frames": safe_get(per_peer_mean_valid, "rtax52_a", 0.0),
            "num_input_frames": safe_get(peer_a, "num_input_frames", 0),
            "informative_windows": safe_get(peer_a, "informative_windows", 0),
            "dense_windows_ge3": safe_get(peer_a, "dense_windows_ge3", 0),
            "full_windows_eq10": safe_get(peer_a, "full_windows_eq10", 0),
            "median_valid_frames": safe_get(peer_a, "median_valid_frames", 0),
        },
        {
            "peer_id": "rtax52_b",
            "backend_name": per_peer_summary.get("rtax52_b", {}).get("backend_name", "unknown"),
            "cbf_count": safe_get(per_peer_summary.get("rtax52_b", {}), "cbf_count", 0),
            "nonzero_windows": safe_get(per_peer_nonzero, "rtax52_b", 0),
            "mean_valid_frames": safe_get(per_peer_mean_valid, "rtax52_b", 0.0),
            "num_input_frames": safe_get(peer_b, "num_input_frames", 0),
            "informative_windows": safe_get(peer_b, "informative_windows", 0),
            "dense_windows_ge3": safe_get(peer_b, "dense_windows_ge3", 0),
            "full_windows_eq10": safe_get(peer_b, "full_windows_eq10", 0),
            "median_valid_frames": safe_get(peer_b, "median_valid_frames", 0),
        },
        {
            "peer_id": "rtax52_c",
            "backend_name": per_peer_summary.get("rtax52_c", {}).get("backend_name", "unknown"),
            "cbf_count": safe_get(per_peer_summary.get("rtax52_c", {}), "cbf_count", 0),
            "nonzero_windows": safe_get(per_peer_nonzero, "rtax52_c", 0),
            "mean_valid_frames": safe_get(per_peer_mean_valid, "rtax52_c", 0.0),
            "num_input_frames": safe_get(peer_c, "num_input_frames", 0),
            "informative_windows": safe_get(peer_c, "informative_windows", 0),
            "dense_windows_ge3": safe_get(peer_c, "dense_windows_ge3", 0),
            "full_windows_eq10": safe_get(peer_c, "full_windows_eq10", 0),
            "median_valid_frames": safe_get(peer_c, "median_valid_frames", 0),
        },
    ]

    df_peers = pd.DataFrame(peers)

    total_cbf = float(safe_get(summary, "cbf_total", 0))
    dominant_peer = None
    dominant_peer_ratio = 0.0
    if total_cbf > 0 and not df_peers.empty:
        idx = df_peers["cbf_count"].astype(float).idxmax()
        dominant_peer = str(df_peers.loc[idx, "peer_id"])
        dominant_peer_ratio = float(df_peers.loc[idx, "cbf_count"]) / total_cbf

    quality_summary = {
        "session_name": args.session_name,
        "summary_json": str(Path(args.summary_json)),
        "fusion_json": str(Path(args.fusion_json)),
        "ndpa": safe_get(summary, "ndpa", 0),
        "cbf_total": safe_get(summary, "cbf_total", 0),
        "num_windows": safe_get(fusion, "num_windows", 0),
        "informative_windows_fused": safe_get(fusion, "informative_windows", 0),
        "fused_shape_per_window": fusion.get("fused_shape_per_window", None),
        "dominant_peer": dominant_peer,
        "dominant_peer_ratio": dominant_peer_ratio,
        "peers": peers,
    }

    json_out = out_dir / f"{args.session_name}_quality_summary.json"
    csv_out = out_dir / f"{args.session_name}_quality_summary.csv"
    png_out = out_dir / f"{args.session_name}_quality_overview.png"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, ensure_ascii=False, indent=2)

    df_peers.to_csv(csv_out, index=False, encoding="utf-8-sig")

    # ---- one-page figure ----
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    # 1. NDPA / CBF total
    ax1 = fig.add_subplot(gs[0, 0])
    vals1 = [float(safe_get(summary, "ndpa", 0)), float(safe_get(summary, "cbf_total", 0))]
    labs1 = ["NDPA", "CBF total"]
    bars1 = ax1.bar(labs1, vals1)
    ax1.set_title("NDPA vs CBF total")
    ymax1 = max(vals1) if vals1 else 1
    ax1.set_ylim(0, ymax1 * 1.2 if ymax1 > 0 else 1)
    for b, v in zip(bars1, vals1):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax1 * 0.03, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=10)

    # 2. CBF by peer
    ax2 = fig.add_subplot(gs[0, 1])
    vals2 = df_peers["cbf_count"].astype(float).tolist()
    labs2 = df_peers["peer_id"].tolist()
    bars2 = ax2.bar(labs2, vals2)
    ax2.set_title("CBF count by peer")
    ymax2 = max(vals2) if vals2 else 1
    ax2.set_ylim(0, ymax2 * 1.2 if ymax2 > 0 else 1)
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax2 * 0.03, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=10)

    # 3. informative windows
    ax3 = fig.add_subplot(gs[0, 2])
    vals3 = [
        float(safe_get(fusion, "num_windows", 0)),
        float(safe_get(fusion, "informative_windows", 0)),
    ]
    labs3 = ["Total windows", "Informative windows"]
    bars3 = ax3.bar(labs3, vals3)
    ax3.set_title("Fused windows")
    ymax3 = max(vals3) if vals3 else 1
    ax3.set_ylim(0, ymax3 * 1.2 if ymax3 > 0 else 1)
    for b, v in zip(bars3, vals3):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax3 * 0.03, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=10)

    # 4. nonzero windows by peer
    ax4 = fig.add_subplot(gs[1, 0])
    vals4 = df_peers["nonzero_windows"].astype(float).tolist()
    labs4 = df_peers["peer_id"].tolist()
    bars4 = ax4.bar(labs4, vals4)
    ax4.set_title("Nonzero windows by peer")
    ymax4 = max(vals4) if vals4 else 1
    ax4.set_ylim(0, ymax4 * 1.2 if ymax4 > 0 else 1)
    for b, v in zip(bars4, vals4):
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax4 * 0.03, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=10)

    # 5. mean valid frames by peer
    ax5 = fig.add_subplot(gs[1, 1])
    vals5 = df_peers["mean_valid_frames"].astype(float).tolist()
    labs5 = df_peers["peer_id"].tolist()
    bars5 = ax5.bar(labs5, vals5)
    ax5.set_title("Mean valid frames by peer")
    ymax5 = max(vals5) if vals5 else 1
    ax5.set_ylim(0, ymax5 * 1.2 if ymax5 > 0 else 1)
    for b, v in zip(bars5, vals5):
        ax5.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax5 * 0.03, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)

    # 6. text panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    text_lines: List[str] = [
        f"Session: {args.session_name}",
        f"CBF total: {safe_get(summary, 'cbf_total', 0)}",
        f"NDPA: {safe_get(summary, 'ndpa', 0)}",
        f"Informative windows: {safe_get(fusion, 'informative_windows', 0)} / {safe_get(fusion, 'num_windows', 0)}",
        f"Dominant peer: {dominant_peer}",
        f"Dominant peer ratio: {dominant_peer_ratio:.3f}",
        "",
        "Per-peer:",
    ]
    for _, row in df_peers.iterrows():
        text_lines.append(
            f"{row['peer_id']} ({row['backend_name']}): "
            f"cbf={row['cbf_count']}, "
            f"info={row['informative_windows']}, "
            f"mean_valid={float(row['mean_valid_frames']):.3f}"
        )
    ax6.text(0, 1, "\n".join(text_lines), va="top", ha="left", fontsize=10)

    fig.suptitle(f"{args.session_name} quality overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(png_out, dpi=170)
    plt.close(fig)

    print(f"[OK] wrote: {json_out}")
    print(f"[OK] wrote: {csv_out}")
    print(f"[OK] wrote: {png_out}")


if __name__ == "__main__":
    main()