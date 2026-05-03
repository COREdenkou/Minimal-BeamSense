#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot numeric columns from a long-tail segment QC CSV file."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def pick_x_column(df):
    candidates = ["segment_id", "segment_idx", "index", "row_id", "window_id"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def sanitize_filename(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


def main():
    parser = argparse.ArgumentParser(description="Generic plotter for longtail_segment_qc.csv")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title-prefix", default="longtail_qc")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.csv)

    x_col = pick_x_column(df)
    if x_col is not None:
        x = df[x_col]
        x_label = x_col
    else:
        x = range(len(df))
        x_label = "row_index"

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    numeric_cols = [c for c in numeric_cols if c != x_col]

    if not numeric_cols:
        print("[WARN] no numeric columns found in CSV.")
        return

    # Create one plot for each numeric column.
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(x, df[col])
        ax.set_title(f"{args.title_prefix} - {col}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"longtail_{sanitize_filename(col)}.png", dpi=160)
        plt.close(fig)

    print(f"[OK] longtail QC plots written to: {out_dir}")


if __name__ == "__main__":
    main()