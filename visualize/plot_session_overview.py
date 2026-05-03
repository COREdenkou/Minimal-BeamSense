#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create overview plots for one processed capture session."""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_bar(labels, values, title, ylabel, out_path, annotate_fmt="{:.0f}", rotation=0):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=rotation)

    ymax = max(values) if len(values) > 0 else 1
    ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03 if ymax > 0 else 0.05,
            annotate_fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_timeline(segments_csv, out_path, total_sec=96):
    df = pd.read_csv(segments_csv)
    for col in ["start_sec", "end_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start_sec", "end_sec"]).sort_values("start_sec")

    fig, ax = plt.subplots(figsize=(11, 2.8))

    for _, row in df.iterrows():
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        duration = max(0, end - start)
        label = str(row["state_label"])

        ax.barh([0], [duration], left=[start], height=0.6, edgecolor="black")
        ax.text(
            start + duration / 2,
            0,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, total_sec)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Session timeline from segments.csv")
    ax.grid(True, axis="x", alpha=0.35)

    tick_max = int(total_sec)
    ax.set_xticks(list(range(0, tick_max + 1, 5)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize one session overview.")
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--fusion-json", required=True)
    parser.add_argument("--segments-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--total-sec", type=float, default=96.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    summary = load_json(args.summary_json)
    fusion = load_json(args.fusion_json)

    # 1. Session timeline
    if Path(args.segments_csv).exists():
        plot_timeline(
            args.segments_csv,
            out_dir / "01_timeline.png",
            total_sec=args.total_sec,
        )

    # 2. NDPA / CBF total
    ndpa = float(summary.get("ndpa", 0))
    cbf_total = float(summary.get("cbf_total", 0))
    save_bar(
        ["NDPA", "CBF total"],
        [ndpa, cbf_total],
        f"{args.session_name} - NDPA vs CBF total",
        "Count",
        out_dir / "02_ndpa_cbf_total.png",
        annotate_fmt="{:.0f}",
    )

    # 3. Per-peer CBF count
    per_peer = summary.get("per_peer", {})
    peer_names = []
    cbf_counts = []
    for peer_name, peer_info in per_peer.items():
        peer_names.append(peer_name)
        cbf_counts.append(float(peer_info.get("cbf_count", 0)))

    if peer_names:
        save_bar(
            peer_names,
            cbf_counts,
            f"{args.session_name} - CBF count by peer",
            "CBF count",
            out_dir / "03_cbf_by_peer.png",
            annotate_fmt="{:.0f}",
        )

    # 4. Informative windows vs total windows
    num_windows = float(fusion.get("num_windows", 0))
    informative = float(fusion.get("informative_windows", 0))
    save_bar(
        ["Total windows", "Informative windows"],
        [num_windows, informative],
        f"{args.session_name} - informative windows",
        "Window count",
        out_dir / "04_informative_windows.png",
        annotate_fmt="{:.0f}",
    )

    # 5. Per-peer nonzero windows
    nonzero = fusion.get("per_peer_nonzero_windows", {})
    if nonzero:
        save_bar(
            list(nonzero.keys()),
            [float(v) for v in nonzero.values()],
            f"{args.session_name} - nonzero windows by peer",
            "Nonzero window count",
            out_dir / "05_nonzero_windows_by_peer.png",
            annotate_fmt="{:.0f}",
        )

    # 6. Per-peer mean valid frames
    mvf = fusion.get("mean_valid_frames_per_peer", {})
    if mvf:
        save_bar(
            list(mvf.keys()),
            [float(v) for v in mvf.values()],
            f"{args.session_name} - mean valid frames per peer",
            "Mean valid frames",
            out_dir / "06_mean_valid_frames_per_peer.png",
            annotate_fmt="{:.3f}",
        )

    print(f"[OK] session overview plots written to: {out_dir}")


if __name__ == "__main__":
    main()