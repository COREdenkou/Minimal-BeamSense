#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot class/window counts from a state master index CSV."""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CANONICAL_ORDER = [
    "empty",
    "single_walking",
    "single_static",
    "single_transition",
    "two_person_active",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def pick_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # Fallback: try fuzzy substring matching.
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def ordered_counts(series):
    counts = series.value_counts().to_dict()
    return [counts.get(k, 0) for k in CANONICAL_ORDER]


def save_bar(labels, values, title, ylabel, out_path, annotate_fmt="{:.0f}", rotation=0):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
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


def main():
    parser = argparse.ArgumentParser(description="Plot state counts from state_master_index.csv")
    parser.add_argument("--state-master-csv", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.state_master_csv)

    session_col = pick_column(
        df,
        ["session_name", "session", "source_session", "session_id"],
    )
    label_col = pick_column(
        df,
        ["state_label", "label", "state", "class_name", "class"],
    )

    if session_col is None:
        raise RuntimeError(
            f"Cannot find session column in state_master_index.csv.\nColumns={list(df.columns)}"
        )
    if label_col is None:
        raise RuntimeError(
            f"Cannot find label column in state_master_index.csv.\nColumns={list(df.columns)}"
        )

    df_session = df[df[session_col].astype(str) == str(args.session_name)].copy()

    if len(df_session) == 0:
        raise RuntimeError(
            f"No rows found for session '{args.session_name}' in column '{session_col}'."
        )

    session_counts = ordered_counts(df_session[label_col].astype(str))
    save_bar(
        CANONICAL_ORDER,
        session_counts,
        f"{args.session_name} - state counts",
        "Window count",
        out_dir / "07_state_counts_session.png",
        annotate_fmt="{:.0f}",
        rotation=20,
    )

    # If a summary JSON is provided, also plot overall and per-session counts.
    if args.summary_json and Path(args.summary_json).exists():
        summary = load_json(args.summary_json)

        state_counts = summary.get("state_counts", {})
        overall_counts = [float(state_counts.get(k, 0)) for k in CANONICAL_ORDER]
        save_bar(
            CANONICAL_ORDER,
            overall_counts,
            "Overall state counts from state_master_index.summary.json",
            "Window count",
            out_dir / "08_state_counts_overall.png",
            annotate_fmt="{:.0f}",
            rotation=20,
        )

        per_session_counts = summary.get("per_session_counts", {})
        if per_session_counts:
            save_bar(
                list(per_session_counts.keys()),
                [float(v) for v in per_session_counts.values()],
                "Per-session counts from state_master_index.summary.json",
                "Window count",
                out_dir / "09_per_session_counts.png",
                annotate_fmt="{:.0f}",
                rotation=20,
            )

    print(f"[OK] state count plots written to: {out_dir}")


if __name__ == "__main__":
    main()