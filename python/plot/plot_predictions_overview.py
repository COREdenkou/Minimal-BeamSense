# -*- coding: utf-8 -*-
"""Generate diagnostic plots from model prediction CSV files."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to predictions.csv")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {"index", "true_label", "pred_label", "pred_conf"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df["correct"] = (df["true_label"] == df["pred_label"]).astype(int)
    df["correct_text"] = df["correct"].map({1: "Correct", 0: "Wrong"})

    # -----------------------------
    # 1) Confidence scatter by sample index
    # -----------------------------
    plt.figure(figsize=(10, 5))
    correct_df = df[df["correct"] == 1]
    wrong_df = df[df["correct"] == 0]

    plt.scatter(correct_df["index"], correct_df["pred_conf"], s=14, alpha=0.7, label="Correct")
    plt.scatter(wrong_df["index"], wrong_df["pred_conf"], s=14, alpha=0.7, label="Wrong")

    plt.xlabel("Sample Index")
    plt.ylabel("Prediction Confidence")
    plt.title("Prediction Confidence by Sample")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pred_conf_scatter.png", dpi=220)
    plt.close()

    # -----------------------------
    # 2) Confidence histogram: Correct vs Wrong
    # -----------------------------
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    plt.hist(correct_df["pred_conf"], bins=bins, alpha=0.7, label="Correct")
    plt.hist(wrong_df["pred_conf"], bins=bins, alpha=0.7, label="Wrong")

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution: Correct vs Wrong")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pred_conf_hist.png", dpi=220)
    plt.close()

    # -----------------------------
    # 3) Per-class correct / wrong bar chart
    # -----------------------------
    summary = (
        df.groupby("true_label")["correct"]
        .agg(total="count", correct="sum")
        .reset_index()
    )
    summary["wrong"] = summary["total"] - summary["correct"]
    summary["accuracy"] = summary["correct"] / summary["total"]

    x = np.arange(len(summary))
    width = 0.6

    plt.figure(figsize=(8, 5))
    plt.bar(x, summary["correct"], width=width, label="Correct")
    plt.bar(x, summary["wrong"], width=width, bottom=summary["correct"], label="Wrong")

    plt.xticks(x, summary["true_label"], rotation=20)
    plt.ylabel("Count")
    plt.xlabel("True Label")
    plt.title("Per-class Correct / Wrong Counts")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "per_class_correct_wrong.png", dpi=220)
    plt.close()

    # Export the per-class summary table.
    summary.to_csv(outdir / "predictions_summary_by_class.csv", index=False, encoding="utf-8-sig")

    print("[OK] wrote:", outdir / "pred_conf_scatter.png")
    print("[OK] wrote:", outdir / "pred_conf_hist.png")
    print("[OK] wrote:", outdir / "per_class_correct_wrong.png")
    print("[OK] wrote:", outdir / "predictions_summary_by_class.csv")


if __name__ == "__main__":
    main()
