#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import matplotlib.pyplot as plt


LABEL_ORDER = [
    "empty",
    "single_walking",
    "single_static",
    "single_transition",
    "two_person_active",
]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def flatten_eval_summary(run_name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_name": run_name,
        "model_name": summary.get("model_name"),
        "accuracy": summary.get("accuracy"),
        "macro_f1": summary.get("macro_f1"),
        "raw_normalize_mode": summary.get("raw_normalize_mode"),
        "norm_normalize_mode": summary.get("norm_normalize_mode"),
        "device": summary.get("device"),
        "test_csv": summary.get("test_csv"),
        "ckpt": summary.get("ckpt"),
    }

    report = summary.get("classification_report", {})
    for label in LABEL_ORDER:
        metrics = report.get(label, {})
        row[f"{label}_precision"] = metrics.get("precision")
        row[f"{label}_recall"] = metrics.get("recall")
        row[f"{label}_f1"] = metrics.get("f1-score")
        row[f"{label}_support"] = metrics.get("support")

    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})
    row["macro_precision"] = macro_avg.get("precision")
    row["macro_recall"] = macro_avg.get("recall")
    row["weighted_precision"] = weighted_avg.get("precision")
    row["weighted_recall"] = weighted_avg.get("recall")
    return row


def main():
    ap = argparse.ArgumentParser(description="Compare multiple eval_summary json files.")
    ap.add_argument("--run-json", nargs="+", required=True,
                    help="One or more eval_summary json paths")
    ap.add_argument("--run-name", nargs="*",
                    help="Optional names for each run; must match number of run-json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    json_paths = [Path(p) for p in args.run_json]
    if args.run_name and len(args.run_name) != len(json_paths):
        raise RuntimeError("If provided, --run-name count must equal --run-json count")

    run_names = args.run_name if args.run_name else [p.parent.name for p in json_paths]

    rows: List[Dict[str, Any]] = []
    summary_bundle: Dict[str, Any] = {"runs": []}

    for run_name, path in zip(run_names, json_paths):
        summary = load_json(path)
        rows.append(flatten_eval_summary(run_name, summary))
        summary_bundle["runs"].append({
            "run_name": run_name,
            "source_json": str(path),
            "accuracy": summary.get("accuracy"),
            "macro_f1": summary.get("macro_f1"),
            "label_order": summary.get("label_order"),
            "model_name": summary.get("model_name"),
            "artifacts": summary.get("artifacts", {}),
        })

    df = pd.DataFrame(rows)

    csv_out = out_dir / "run_compare_table.csv"
    json_out = out_dir / "run_compare_summary.json"
    png_main = out_dir / "run_compare_metrics.png"
    png_class = out_dir / "run_compare_per_class_f1.png"

    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_bundle, f, ensure_ascii=False, indent=2)

    # 1) overall metrics plot
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(df))
    width = 0.38

    acc_vals = pd.to_numeric(df["accuracy"], errors="coerce").fillna(0).tolist()
    f1_vals = pd.to_numeric(df["macro_f1"], errors="coerce").fillna(0).tolist()

    ax.bar([i - width/2 for i in x], acc_vals, width=width, label="accuracy")
    ax.bar([i + width/2 for i in x], f1_vals, width=width, label="macro_f1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["run_name"].astype(str).tolist(), rotation=20)
    ax.set_ylim(0, max(acc_vals + f1_vals) * 1.25 if len(acc_vals + f1_vals) > 0 else 1)
    ax.set_ylabel("metric value")
    ax.set_title("Run comparison: accuracy vs macro_f1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_main, dpi=170)
    plt.close(fig)

    # 2) per-class f1 plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label in LABEL_ORDER:
        col = f"{label}_f1"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).tolist()
            ax.plot(df["run_name"].astype(str).tolist(), vals, marker="o", label=label)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Run comparison: per-class F1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_class, dpi=170)
    plt.close(fig)

    print(f"[OK] wrote: {csv_out}")
    print(f"[OK] wrote: {json_out}")
    print(f"[OK] wrote: {png_main}")
    print(f"[OK] wrote: {png_class}")


if __name__ == "__main__":
    main()