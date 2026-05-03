# plot_train_accuracy_blue_orange.py
"""Plot training and validation accuracy from a Keras history CSV."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to training history CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Training / Validation Accuracy")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    epoch_col = find_col(df, ["epoch"])
    train_acc_col = find_col(df, ["accuracy", "acc", "train_accuracy", "train_acc"])
    val_acc_col = find_col(df, ["val_accuracy", "val_acc"])

    if train_acc_col is None or val_acc_col is None:
        raise RuntimeError(
            f"Cannot find accuracy columns. Found columns: {list(df.columns)}"
        )

    if epoch_col is None:
        x = list(range(1, len(df) + 1))
    else:
        x = df[epoch_col].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, df[train_acc_col], label="Train Accuracy", color="tab:blue", linewidth=2)
    ax.plot(x, df[val_acc_col], label="Val Accuracy", color="tab:orange", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
