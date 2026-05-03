#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the blind-inference timeline with interval-level majority voting."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CLASS_ORDER = ["empty", "one_person", "two_person"]
CLASS_TO_Y = {k: i for i, k in enumerate(CLASS_ORDER)}
CLASS_COLORS = {
    "empty": "blue",
    "one_person": "green",
    "two_person": "orange",
}


def main():
    ap = argparse.ArgumentParser(description="Plot sample predictions and majority-voted intervals.")
    ap.add_argument("--sample-csv", required=True)
    ap.add_argument("--interval-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Inference Timeline with 5 s Majority Voting")
    args = ap.parse_args()

    sample_csv = Path(args.sample_csv)
    interval_csv = Path(args.interval_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(sample_csv)
    intervals = pd.read_csv(interval_csv)

    required = {"center_sec", "pred_label"}
    missing = required - set(samples.columns)
    if missing:
        raise RuntimeError(f"Missing columns in sample csv: {missing}")

    samples = samples[samples["pred_label"].isin(CLASS_ORDER)].copy()
    samples["y"] = samples["pred_label"].map(CLASS_TO_Y)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Draw the majority-voted 5 s intervals as light background spans.
    label_y = len(CLASS_ORDER) - 0.12

    for _, r in intervals.iterrows():
        lab = str(r["majority_label"])
        color = CLASS_COLORS.get(lab, "blue")
        start = float(r["start_sec"])
        end = float(r["end_sec"])
        ratio = float(r.get("majority_ratio", 0.0))

        ax.axvspan(start, end, color=color, alpha=0.08)

        mid = (start + end) / 2
        ax.text(
            mid,
            label_y,
            f"{lab}\nr={ratio:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )

    # Draw sample-level predictions.
    for lab in CLASS_ORDER:
        d = samples[samples["pred_label"] == lab]
        if len(d) == 0:
            continue
        ax.scatter(
            d["center_sec"],
            d["y"],
            s=8,
            marker=".",
            color=CLASS_COLORS[lab],
            alpha=0.75,
            label=lab,
        )

    # Mark interval boundaries.
    if len(intervals):
        borders = sorted(set(intervals["start_sec"].astype(float).tolist() + intervals["end_sec"].astype(float).tolist()))
        for x in borders:
            ax.axvline(x, color="black", alpha=0.12, linewidth=0.8)

    ax.set_yticks(list(range(len(CLASS_ORDER))))
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_ylim(-0.5, len(CLASS_ORDER) + 0.35)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted class")
    ax.set_title(args.title)

    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(True, axis="y", alpha=0.15)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("[OK] wrote:", out_path)


if __name__ == "__main__":
    main()
