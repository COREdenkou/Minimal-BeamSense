# -*- coding: utf-8 -*-
"""
Plot final metrics table from Keras eval_summary.json.

Expected metrics:
- overall accuracy
- macro-F1
- per-class F1-score

Outputs:
- final_metrics_table.png
- final_metrics_table.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def safe_get(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(x):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def load_metrics(eval_json: Path):
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = data.get("classification_report", {})

    # Overall accuracy: support multiple possible schemas
    accuracy = data.get("accuracy", None)
    if accuracy is None:
        accuracy = report.get("accuracy", None)

    macro_f1 = data.get("macro_f1", None)
    if macro_f1 is None:
        macro_f1 = safe_get(report, ["macro avg", "f1-score"])

    weighted_f1 = data.get("weighted_f1", None)
    if weighted_f1 is None:
        weighted_f1 = safe_get(report, ["weighted avg", "f1-score"])

    rows = []

    rows.append({
        "Category": "Overall",
        "Metric": "Accuracy",
        "Value": accuracy,
    })

    rows.append({
        "Category": "Overall",
        "Metric": "Macro-F1",
        "Value": macro_f1,
    })

    if weighted_f1 is not None:
        rows.append({
            "Category": "Overall",
            "Metric": "Weighted-F1",
            "Value": weighted_f1,
        })

    # Per-class F1
    ignored = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}
    for label, item in report.items():
        if label in ignored:
            continue
        if not isinstance(item, dict):
            continue
        if "f1-score" not in item:
            continue

        rows.append({
            "Category": "Per-class",
            "Metric": f"{label} F1",
            "Value": item.get("f1-score"),
        })

    return pd.DataFrame(rows)


def plot_table(df: pd.DataFrame, out_png: Path, title: str):
    show_df = df.copy()
    show_df["Value"] = show_df["Value"].apply(fmt)

    fig_height = max(2.8, 0.45 * len(show_df) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)

    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f77b4")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#eef5fb")
            else:
                cell.set_facecolor("#ffffff")

    ax.set_title(title, fontsize=16, pad=18)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to eval_summary.json")
    parser.add_argument("--out-png", required=True, help="Output table PNG path")
    parser.add_argument("--out-csv", default=None, help="Optional output CSV path")
    parser.add_argument("--title", default="Final Classification Metrics")
    args = parser.parse_args()

    eval_json = Path(args.json)
    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv) if args.out_csv else out_png.with_suffix(".csv")

    df = load_metrics(eval_json)

    if df.empty:
        raise RuntimeError(f"No metrics found in {eval_json}")

    # Save numeric CSV
    csv_df = df.copy()
    csv_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Plot PNG
    plot_table(df, out_png, args.title)

    print("[OK] wrote:", out_png)
    print("[OK] wrote:", out_csv)
    print(df)


if __name__ == "__main__":
    main()
