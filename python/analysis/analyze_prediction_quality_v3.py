#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt


CANDIDATES = {
    "correct": ["is_correct", "correct"],
    "pred_conf": ["pred_conf", "confidence", "max_prob", "top1_prob"],
    "valid_peer_count": ["valid_peer_count"],
    "total_valid_frames": ["total_valid_frames", "valid_frames_total"],
    "a_valid": ["rtax52_a_valid_frames", "a_valid_frames", "peer_a_valid_frames"],
    "b_valid": ["rtax52_b_valid_frames", "b_valid_frames", "peer_b_valid_frames"],
    "c_valid": ["rtax52_c_valid_frames", "c_valid_frames", "peer_c_valid_frames"],
    "dominant_peer_ratio": ["dominant_peer_ratio"],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, keys: List[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in df.columns:
            return k
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    for c in df.columns:
        cl = c.lower()
        for k in keys:
            if k.lower() in cl:
                return c
    return None


def safe_stats(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare quality metrics between correct and wrong predictions.")
    ap.add_argument("--aligned-pred-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    in_csv = Path(args.aligned_pred_csv)
    out_dir = Path(args.out_dir)

    print(f"[INFO] input_csv: {in_csv}")
    print(f"[INFO] out_dir   : {out_dir}")

    if not in_csv.exists():
        raise RuntimeError(f"Input CSV not found: {in_csv}")

    ensure_dir(out_dir)

    df = pd.read_csv(in_csv)
    print(f"[INFO] loaded rows: {len(df)}")
    print(f"[INFO] columns    : {list(df.columns)}")

    col_correct = pick_col(df, CANDIDATES["correct"])

    # If there is no explicit correctness column, infer it from ground-truth and predicted labels.
    if col_correct is None:
        gt_col = pick_col(df, ["gt_label", "true_label", "state_label"])
        pred_col = pick_col(df, ["pred_label", "pred"])
        if gt_col is None or pred_col is None:
            raise RuntimeError(
                "Cannot find correctness column and cannot infer it from gt/pred labels.\n"
                f"Columns={list(df.columns)}"
            )
        df["is_correct"] = (df[gt_col].astype(str) == df[pred_col].astype(str)).astype(int)
        col_correct = "is_correct"

    df[col_correct] = pd.to_numeric(df[col_correct], errors="coerce").fillna(0).astype(int)

    feature_cols = {
        key: pick_col(df, candidates)
        for key, candidates in CANDIDATES.items()
        if key != "correct"
    }

    print(f"[INFO] correctness_col: {col_correct}")
    print(f"[INFO] feature_cols    : {feature_cols}")

    rows = []
    summary_json: Dict[str, Any] = {
        "input_csv": str(in_csv),
        "out_dir": str(out_dir),
        "n_rows": int(len(df)),
        "correctness_col": col_correct,
        "feature_cols": feature_cols,
        "metrics": {},
    }

    use_metrics = []
    for name, col in feature_cols.items():
        if col is not None:
            use_metrics.append((name, col))

    if len(use_metrics) == 0:
        print("[WARN] no usable quality metric columns found")
    else:
        print(f"[INFO] usable metrics : {[m for m, _ in use_metrics]}")

    for metric_name, metric_col in use_metrics:
        correct_vals = df.loc[df[col_correct] == 1, metric_col]
        wrong_vals = df.loc[df[col_correct] == 0, metric_col]

        correct_stats = safe_stats(correct_vals)
        wrong_stats = safe_stats(wrong_vals)

        summary_json["metrics"][metric_name] = {
            "column": metric_col,
            "correct": correct_stats,
            "wrong": wrong_stats,
        }

        rows.append({
            "metric": metric_name,
            "column": metric_col,
            "correct_n": correct_stats["n"],
            "correct_mean": correct_stats["mean"],
            "correct_median": correct_stats["median"],
            "correct_std": correct_stats["std"],
            "wrong_n": wrong_stats["n"],
            "wrong_mean": wrong_stats["mean"],
            "wrong_median": wrong_stats["median"],
            "wrong_std": wrong_stats["std"],
        })

    summary_csv = out_dir / "quality_vs_correctness.csv"
    summary_json_path = out_dir / "quality_vs_correctness_summary.json"
    plot_path = out_dir / "quality_vs_correctness.png"

    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {summary_csv}")
    print(f"[OK] wrote: {summary_json_path}")

    if rows:
        metrics = [r["metric"] for r in rows]
        correct_means = [r["correct_mean"] if r["correct_mean"] is not None else 0 for r in rows]
        wrong_means = [r["wrong_mean"] if r["wrong_mean"] is not None else 0 for r in rows]

        x = range(len(metrics))
        width = 0.38

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar([i - width / 2 for i in x], correct_means, width=width, label="correct mean")
        ax.bar([i + width / 2 for i in x], wrong_means, width=width, label="wrong mean")

        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics, rotation=20)
        ax.set_ylabel("value")
        ax.set_title("Quality metrics: correct vs wrong predictions")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=170)
        plt.close(fig)

        print(f"[OK] wrote: {plot_path}")
    else:
        print("[WARN] no rows plotted because no usable metrics found")


if __name__ == "__main__":
    main()