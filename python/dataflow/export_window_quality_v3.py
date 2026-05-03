#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


QUALITY_COLS = [
    "pred_conf",
    "valid_peer_count_v3",
    "total_valid_frames_v3",
    "rtax52_a_valid_frames_v3",
    "rtax52_b_valid_frames_v3",
    "rtax52_c_valid_frames_v3",
    "dominant_peer_ratio",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser(description="Export window-quality summaries from merged v3 prediction table.")
    ap.add_argument("--merged-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    merged_csv = Path(args.merged_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not merged_csv.exists():
        raise RuntimeError(f"merged csv not found: {merged_csv}")

    df = pd.read_csv(merged_csv)

    required = ["session_name", "state_label", "pred_label", "is_correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"merged csv missing required columns: {missing}")

    qcols = existing_cols(df, QUALITY_COLS)

    # -------------------------------------------------
    # 1) overall correct vs wrong
    # -------------------------------------------------
    overall_rows = []
    for metric in qcols:
        correct_mean = safe_mean(df.loc[df["is_correct"] == 1, metric])
        wrong_mean = safe_mean(df.loc[df["is_correct"] == 0, metric])
        overall_rows.append({
            "metric": metric,
            "correct_mean": correct_mean,
            "wrong_mean": wrong_mean,
        })

    overall_df = pd.DataFrame(overall_rows)
    overall_csv = out_dir / "window_quality_overall.csv"
    overall_df.to_csv(overall_csv, index=False, encoding="utf-8-sig")

    # -------------------------------------------------
    # 2) by session
    # -------------------------------------------------
    agg_spec = {
        "n_windows": ("is_correct", "size"),
        "acc": ("is_correct", "mean"),
    }
    for c in qcols:
        agg_spec[f"{c}_mean"] = (c, "mean")

    by_session = (
        df.groupby("session_name", dropna=False)
          .agg(**agg_spec)
          .reset_index()
          .sort_values(["acc", "n_windows"], ascending=[True, False])
    )
    by_session_csv = out_dir / "window_quality_by_session.csv"
    by_session.to_csv(by_session_csv, index=False, encoding="utf-8-sig")

    # -------------------------------------------------
    # 3) by class
    # -------------------------------------------------
    by_class = (
        df.groupby("state_label", dropna=False)
          .agg(**agg_spec)
          .reset_index()
          .sort_values("acc", ascending=True)
    )
    by_class_csv = out_dir / "window_quality_by_class.csv"
    by_class.to_csv(by_class_csv, index=False, encoding="utf-8-sig")

    # -------------------------------------------------
    # 4) by session x class
    # -------------------------------------------------
    by_session_class = (
        df.groupby(["session_name", "state_label"], dropna=False)
          .agg(**agg_spec)
          .reset_index()
          .sort_values(["session_name", "state_label"])
    )
    by_session_class_csv = out_dir / "window_quality_by_session_and_class.csv"
    by_session_class.to_csv(by_session_class_csv, index=False, encoding="utf-8-sig")

    # -------------------------------------------------
    # 5) summary json
    # -------------------------------------------------
    summary = {
        "input_csv": str(merged_csv),
        "n_rows": int(len(df)),
        "n_sessions": int(df["session_name"].nunique()),
        "n_classes": int(df["state_label"].nunique()),
        "quality_columns": qcols,
        "overall_accuracy": float(df["is_correct"].mean()),
        "files": {
            "window_quality_overall_csv": str(overall_csv),
            "window_quality_by_session_csv": str(by_session_csv),
            "window_quality_by_class_csv": str(by_class_csv),
            "window_quality_by_session_and_class_csv": str(by_session_class_csv),
        },
    }
    summary_json = out_dir / "window_quality_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------
    # 6) plots
    # -------------------------------------------------
    if len(overall_df) > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = range(len(overall_df))
        width = 0.38
        ax.bar([i - width/2 for i in x], overall_df["correct_mean"].fillna(0), width=width, label="correct mean")
        ax.bar([i + width/2 for i in x], overall_df["wrong_mean"].fillna(0), width=width, label="wrong mean")
        ax.set_xticks(list(x))
        ax.set_xticklabels(overall_df["metric"].tolist(), rotation=20)
        ax.set_title("Window quality: correct vs wrong")
        ax.set_ylabel("value")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "window_quality_overall.png", dpi=170)
        plt.close(fig)

    if len(by_session) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(by_session["session_name"].astype(str), by_session["acc"])
        ax.set_title("Per-session accuracy from merged v3 table")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "window_quality_by_session.png", dpi=170)
        plt.close(fig)

    if len(by_class) > 0:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(by_class["state_label"].astype(str), by_class["acc"])
        ax.set_title("Per-class accuracy from merged v3 table")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "window_quality_by_class.png", dpi=170)
        plt.close(fig)

    print(f"[OK] wrote: {overall_csv}")
    print(f"[OK] wrote: {by_session_csv}")
    print(f"[OK] wrote: {by_class_csv}")
    print(f"[OK] wrote: {by_session_class_csv}")
    print(f"[OK] wrote: {summary_json}")
    print(f"[INFO] n_rows={len(df)}, n_sessions={df['session_name'].nunique()}, n_classes={df['state_label'].nunique()}")


if __name__ == "__main__":
    main()