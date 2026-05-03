# -*- coding: utf-8 -*-
"""Plot prediction labels over sample index for quick trend inspection."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LABEL_ORDER = ["empty", "one_person", "two_person"]
LABEL_TO_Y = {k: i for i, k in enumerate(LABEL_ORDER)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to predictions.csv")
    parser.add_argument("--out", required=True, help="Output png path")
    parser.add_argument("--title", default="Prediction Trend by Sample Index")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {"index", "true_label", "pred_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # Keep only the known labels used by the current model.
    df = df[df["true_label"].isin(LABEL_ORDER) & df["pred_label"].isin(LABEL_ORDER)].copy()

    # Sort samples by index.
    df = df.sort_values("index").reset_index(drop=True)

    # Map labels to y-axis positions.
    df["pred_y"] = df["pred_label"].map(LABEL_TO_Y).astype(float)

    # Mark whether each prediction is correct.
    df["correct"] = df["true_label"] == df["pred_label"]

    plt.figure(figsize=(14, 5))

    # Correct predictions.
    d_ok = df[df["correct"]]
    if len(d_ok) > 0:
        plt.scatter(
            d_ok["index"],
            d_ok["pred_y"],
            s=10,
            marker=".",
            color="tab:blue",
            label="Correct"
        )

    # Wrong predictions.
    d_ng = df[~df["correct"]]
    if len(d_ng) > 0:
        plt.scatter(
            d_ng["index"],
            d_ng["pred_y"],
            s=5,
            marker=".",
            color="tab:orange",
            label="Wrong"
        )

    plt.yticks(
        ticks=list(range(len(LABEL_ORDER))),
        labels=LABEL_ORDER
    )
    plt.ylim(-0.5, len(LABEL_ORDER) - 0.5)

    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Class")
    plt.title(args.title)
    plt.grid(True, axis="x", alpha=0.25)
    plt.grid(True, axis="y", alpha=0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
