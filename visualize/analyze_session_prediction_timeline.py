#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Align session-level predictions with segment labels and plot timeline diagnostics."""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


LABEL_ORDER = [
    "empty",
    "single_walking",
    "single_static",
    "single_transition",
    "two_person_active",
]
LABEL_TO_Y = {k: i for i, k in enumerate(LABEL_ORDER)}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="eval/predictions_v2.csv")
    ap.add_argument("--state-test-csv", required=True, help="split_excl_001_004/state_test_v2.csv")
    ap.add_argument("--segments-csv", required=True, help="segments/session_segments.csv")
    ap.add_argument("--session-name", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    pred = pd.read_csv(args.pred_csv)
    test = pd.read_csv(args.state_test_csv)
    seg = pd.read_csv(args.segments_csv)

    required_pred_cols = {"index", "true_label", "pred_label"}
    missing_pred = required_pred_cols - set(pred.columns)
    if missing_pred:
        raise RuntimeError(f"predictions_v2.csv missing columns: {missing_pred}")

    required_test_cols = {"session_name", "state_label", "time_center_sec"}
    missing_test = required_test_cols - set(test.columns)
    if missing_test:
        raise RuntimeError(f"state_test_v2.csv missing columns: {missing_test}")

    # Align prediction indices with state_test row indices.
    test = test.reset_index().rename(columns={"index": "row_index"})
    pred = pred.rename(columns={"index": "row_index"})

    df = test.merge(
        pred[["row_index", "true_label", "pred_label", *([c for c in ["pred_conf"] if c in pred.columns])]],
        on="row_index",
        how="inner",
    )

    df = df[df["session_name"].astype(str) == args.session_name].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for session {args.session_name} after merge.")

    df["time_center_sec"] = pd.to_numeric(df["time_center_sec"], errors="coerce")
    df = df.dropna(subset=["time_center_sec"]).sort_values("time_center_sec").reset_index(drop=True)

    # Use state_label from state_test as the reference label.
    df["gt_label"] = df["state_label"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str)
    df["is_correct"] = (df["gt_label"] == df["pred_label"]).astype(int)

    seg["start_sec"] = pd.to_numeric(seg["start_sec"], errors="coerce")
    seg["end_sec"] = pd.to_numeric(seg["end_sec"], errors="coerce")
    seg = seg.dropna(subset=["start_sec", "end_sec"]).sort_values("start_sec")

    aligned_csv = out_dir / f"{args.session_name}_aligned_predictions.csv"
    df.to_csv(aligned_csv, index=False, encoding="utf-8-sig")

    # segment-wise summary
    segment_summary = (
        df.groupby("gt_label")
          .agg(
              n=("is_correct", "size"),
              acc=("is_correct", "mean"),
              majority_pred=("pred_label", lambda s: s.value_counts().index[0]),
              mean_conf=("pred_conf", "mean") if "pred_conf" in df.columns else ("is_correct", "mean"),
          )
          .reset_index()
    )
    segment_summary.to_csv(out_dir / f"{args.session_name}_segment_summary.csv", index=False, encoding="utf-8-sig")

    # within-session confusion
    conf = pd.crosstab(df["gt_label"], df["pred_label"], dropna=False)
    conf.to_csv(out_dir / f"{args.session_name}_confusion_within_session.csv", encoding="utf-8-sig")

    # 1) Timeline: reference labels vs predictions
    fig, ax = plt.subplots(figsize=(13, 4.5))
    gt_y = [LABEL_TO_Y.get(x, -1) for x in df["gt_label"]]
    pred_y = [LABEL_TO_Y.get(x, -1) for x in df["pred_label"]]

    ax.scatter(df["time_center_sec"], gt_y, s=22, label="ground truth", alpha=0.9)
    ax.scatter(df["time_center_sec"], pred_y, s=12, label="prediction", alpha=0.7)

    for _, row in seg.iterrows():
        ax.axvspan(row["start_sec"], row["end_sec"], alpha=0.08)
        ax.text((row["start_sec"] + row["end_sec"]) / 2, 4.35, str(row["state_label"]),
                ha="center", va="bottom", fontsize=9)

    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Class")
    ax.set_title(f"{args.session_name} timeline: ground truth vs prediction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.session_name}_timeline_true_pred.png", dpi=170)
    plt.close(fig)

    # 2) Segment-wise accuracy
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = segment_summary["gt_label"].astype(str).tolist()
    y = segment_summary["acc"].fillna(0).tolist()
    bars = ax.bar(x, y)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{args.session_name} segment-wise accuracy")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.session_name}_segment_accuracy.png", dpi=170)
    plt.close(fig)

    # 3) Prediction confidence over time
    if "pred_conf" in df.columns:
        fig, ax = plt.subplots(figsize=(13, 3.8))
        ax.plot(df["time_center_sec"], pd.to_numeric(df["pred_conf"], errors="coerce"))
        for _, row in seg.iterrows():
            ax.axvspan(row["start_sec"], row["end_sec"], alpha=0.06)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("pred_conf")
        ax.set_title(f"{args.session_name} confidence over time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.session_name}_confidence_over_time.png", dpi=170)
        plt.close(fig)

    print(f"[OK] outputs written to: {out_dir}")
    print(f"[OK] aligned csv: {aligned_csv}")


if __name__ == "__main__":
    main()