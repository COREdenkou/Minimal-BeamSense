# -*- coding: utf-8 -*-
"""
Summarize one Keras BeamSense-style run.

Inputs:
- eval_summary.json
- confusion_matrix.csv
- train_history.csv
- optional split_summary json

Outputs:
- run_summary_text.txt
- run_summary_metrics.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    eval_dir = model_dir / "eval"
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "summary_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_json = eval_dir / "eval_summary.json"
    cm_csv = eval_dir / "confusion_matrix.csv"
    hist_csv = model_dir / "train_history.csv"

    if not eval_json.is_file():
        raise FileNotFoundError(f"Missing eval_summary.json: {eval_json}")
    if not cm_csv.is_file():
        raise FileNotFoundError(f"Missing confusion_matrix.csv: {cm_csv}")
    if not hist_csv.is_file():
        raise FileNotFoundError(f"Missing train_history.csv: {hist_csv}")

    with open(eval_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    cm = pd.read_csv(cm_csv, index_col=0)
    hist = pd.read_csv(hist_csv)

    labels = list(cm.index)

    # Eval metrics: handle several possible schema variants
    accuracy = eval_data.get("accuracy", None)
    macro_f1 = eval_data.get("macro_f1", None)
    weighted_f1 = eval_data.get("weighted_f1", None)

    report = eval_data.get("classification_report", {})
    if macro_f1 is None:
        macro_f1 = safe_get(report, ["macro avg", "f1-score"])
    if weighted_f1 is None:
        weighted_f1 = safe_get(report, ["weighted avg", "f1-score"])

    # History metrics
    last = hist.iloc[-1].to_dict()
    best_val_acc_idx = hist["val_accuracy"].idxmax() if "val_accuracy" in hist.columns else None
    best_val_loss_idx = hist["val_loss"].idxmin() if "val_loss" in hist.columns else None

    best_val_acc = None
    best_val_acc_epoch = None
    if best_val_acc_idx is not None:
        best_val_acc = float(hist.loc[best_val_acc_idx, "val_accuracy"])
        best_val_acc_epoch = int(hist.loc[best_val_acc_idx, "epoch"]) if "epoch" in hist.columns else int(best_val_acc_idx + 1)

    best_val_loss = None
    best_val_loss_epoch = None
    if best_val_loss_idx is not None:
        best_val_loss = float(hist.loc[best_val_loss_idx, "val_loss"])
        best_val_loss_epoch = int(hist.loc[best_val_loss_idx, "epoch"]) if "epoch" in hist.columns else int(best_val_loss_idx + 1)

    final_train_acc = float(last["accuracy"]) if "accuracy" in last else None
    final_val_acc = float(last["val_accuracy"]) if "val_accuracy" in last else None
    final_train_loss = float(last["loss"]) if "loss" in last else None
    final_val_loss = float(last["val_loss"]) if "val_loss" in last else None

    acc_gap = None
    if final_train_acc is not None and final_val_acc is not None:
        acc_gap = final_train_acc - final_val_acc

    loss_gap = None
    if final_train_loss is not None and final_val_loss is not None:
        loss_gap = final_val_loss - final_train_loss

    # Confusion matrix recalls and major confusions
    cm_values = cm.astype(float)
    per_class_recall = {}
    for lab in labels:
        per_class_recall[lab] = float(cm_values.loc[lab, lab])

    confusions = {}
    for true_lab in labels:
        row = cm_values.loc[true_lab]
        off_diag = row.drop(index=true_lab)
        if len(off_diag) > 0:
            pred_lab = off_diag.idxmax()
            confusions[true_lab] = {
                "most_confused_as": str(pred_lab),
                "ratio": float(off_diag.loc[pred_lab]),
            }

    summary = {
        "model_dir": str(model_dir),
        "labels": labels,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "best_val_accuracy": best_val_acc,
        "best_val_accuracy_epoch": best_val_acc_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_accuracy_gap_train_minus_val": acc_gap,
        "final_loss_gap_val_minus_train": loss_gap,
        "per_class_recall_from_confusion_matrix": per_class_recall,
        "main_confusions": confusions,
    }

    with open(out_dir / "run_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("=== Keras BeamSense-style Run Summary ===")
    lines.append(f"Model dir: {model_dir}")
    lines.append("")
    lines.append("[Test metrics]")
    lines.append(f"Accuracy     : {accuracy}")
    lines.append(f"Macro-F1     : {macro_f1}")
    lines.append(f"Weighted-F1  : {weighted_f1}")
    lines.append("")
    lines.append("[Training dynamics]")
    lines.append(f"Best val accuracy : {best_val_acc} @ epoch {best_val_acc_epoch}")
    lines.append(f"Best val loss     : {best_val_loss} @ epoch {best_val_loss_epoch}")
    lines.append(f"Final train acc   : {final_train_acc}")
    lines.append(f"Final val acc     : {final_val_acc}")
    lines.append(f"Final train loss  : {final_train_loss}")
    lines.append(f"Final val loss    : {final_val_loss}")
    lines.append(f"Final acc gap     : {acc_gap}")
    lines.append(f"Final loss gap    : {loss_gap}")
    lines.append("")
    lines.append("[Per-class recall from normalized confusion matrix]")
    for lab, val in per_class_recall.items():
        lines.append(f"{lab:>12s}: {val:.3f}")
    lines.append("")
    lines.append("[Main confusion per true class]")
    for lab, item in confusions.items():
        lines.append(f"{lab:>12s} -> {item['most_confused_as']}: {item['ratio']:.3f}")

    text = "\n".join(lines)
    (out_dir / "run_summary_text.txt").write_text(text, encoding="utf-8")

    print(text)
    print(f"\n[OK] wrote: {out_dir}")


if __name__ == "__main__":
    main()