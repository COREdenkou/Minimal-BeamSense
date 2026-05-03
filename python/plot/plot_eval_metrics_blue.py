# plot_eval_metrics_blue.py
"""Plot scalar evaluation metrics from eval_summary.json."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to eval_summary.json")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Evaluation Metrics")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = [
        ("accuracy", data.get("accuracy")),
        ("macro_f1", data.get("macro_f1")),
        ("weighted_f1", data.get("weighted_f1")),
        ("macro_precision", data.get("macro_precision")),
        ("macro_recall", data.get("macro_recall")),
    ]

    metrics = [(k, v) for k, v in candidates if isinstance(v, (int, float))]
    if not metrics:
        raise RuntimeError(f"No plottable metrics found in: {args.json}")

    names = [k for k, _ in metrics]
    values = [v for _, v in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color="tab:blue")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
