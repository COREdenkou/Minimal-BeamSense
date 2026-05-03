# -*- coding: utf-8 -*-
"""
Plot normalised confusion matrix from CSV.
- Fixed value scale: 0 ~ 1
- Blue colormap
- Show only one number in each cell
- Keep 3 decimals
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_matrix(csv_path: Path):
    """
    Load confusion matrix from CSV.
    Supports:
    1) first column as row labels
    2) plain numeric matrix without index column
    """
    # Try reading with first column as index
    df = pd.read_csv(csv_path, index_col=0)

    # If column count mismatches or content looks wrong, fallback
    if df.shape[0] == 0 or df.shape[1] == 0:
        df = pd.read_csv(csv_path)

    # If all columns are unnamed numeric-like due to no index, reload plainly
    if any(str(c).startswith("Unnamed") for c in df.columns):
        df = pd.read_csv(csv_path)

    # Try to convert all entries to numeric
    try:
        mat = df.astype(float).values
        row_labels = [str(x) for x in df.index.tolist()]
        col_labels = [str(x) for x in df.columns.tolist()]
    except Exception:
        df2 = pd.read_csv(csv_path)
        mat = df2.astype(float).values
        row_labels = [str(i) for i in range(mat.shape[0])]
        col_labels = [str(i) for i in range(mat.shape[1])]

    # If labels are default integer index but columns are meaningful, use columns for both axes
    if row_labels == [str(i) for i in range(len(row_labels))] and len(col_labels) == mat.shape[0]:
        row_labels = col_labels.copy()

    return mat, row_labels, col_labels


def plot_confusion_matrix(
    cm: np.ndarray,
    row_labels,
    col_labels,
    title: str,
    out_path: Path,
    figsize=(8, 7),
    dpi=200
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Fixed normalised scale
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)

    # Axis ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=12)
    ax.set_yticklabels(row_labels, fontsize=12)

    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)
    ax.set_title(title, fontsize=16, pad=16)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value", rotation=270, labelpad=20, fontsize=12)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=11)

    # Annotate each cell with one value only, 3 decimals
    threshold = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_color = "white" if val >= threshold else "black"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                color=text_color, fontsize=12
            )

    # Thin white gridlines between cells
    ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot normalised confusion matrix from CSV")
    parser.add_argument("--csv", required=True, help="Path to confusion matrix CSV")
    parser.add_argument("--out", required=True, help="Output image path, e.g. xxx.png")
    parser.add_argument("--title", default="Normalised Confusion Matrix - Session Split",
                        help="Figure title")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    cm, row_labels, col_labels = load_matrix(csv_path)

    # Safety check
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape={cm.shape}")

    plot_confusion_matrix(
        cm=cm,
        row_labels=row_labels,
        col_labels=col_labels,
        title=args.title,
        out_path=out_path
    )


if __name__ == "__main__":
    main()