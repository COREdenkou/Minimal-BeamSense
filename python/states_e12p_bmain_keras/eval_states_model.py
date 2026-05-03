#!/usr/bin/env python3
"""Evaluate a trained three-state Minimal BeamSense model and save predictions/metrics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from dataGenerator_states import DataGenerator, DEFAULT_STATE_ORDER


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained three-state Minimal BeamSense baseline model."
    )
    p.add_argument(
        "--test-csv",
        default=r"<PROJECT_ROOT>\state_training\state_test.csv",
        help="Path to state_test.csv",
    )
    p.add_argument(
        "--model-path",
        default=r"<PROJECT_ROOT>\state_training\models\states_baseline\states_baseline_best.keras",
        help="Path to trained Keras model",
    )
    p.add_argument(
        "--label-map-json",
        default=r"<PROJECT_ROOT>\state_training\models\states_baseline\label_map.json",
        help="Path to saved label_map.json",
    )
    p.add_argument(
        "--out-dir",
        default=r"<PROJECT_ROOT>\state_training\models\states_baseline\eval",
        help="Directory to save evaluation outputs",
    )
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def load_label_order(label_map_json: Path):
    with open(label_map_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    label_order = obj.get("label_order", [])
    if not label_order:
        raise RuntimeError(f"label_order missing in {label_map_json}")
    return label_order


def plot_confusion_matrix(cm, labels, out_png: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual",
        xlabel="Predicted",
        title="Normalised Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    test_csv = Path(args.test_csv)
    model_path = Path(args.model_path)
    label_map_json = Path(args.label_map_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_csv.is_file():
        raise FileNotFoundError(f"test_csv not found: {test_csv}")
    if not model_path.is_file():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not label_map_json.is_file():
        raise FileNotFoundError(f"label_map_json not found: {label_map_json}")

    label_order = load_label_order(label_map_json)

    test_gen = DataGenerator(
        dataset_csv=str(test_csv),
        batchsize=args.batch_size,
        shuffle=False,
        to_categorical=True,
        label_order=label_order,
        normalize_by_180=True,
    )

    model = load_model(model_path)

    loss, acc = model.evaluate(test_gen, verbose=1)
    probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Generator keeps order stable when shuffle=False
    y_true_labels = [test_gen.labels[i] for i in test_gen.indexes[: len(y_pred)]]
    y_true = np.array([test_gen.label_to_index[x] for x in y_true_labels], dtype=int)

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_order,
        output_dict=True,
        zero_division=0,
    )

    # Save confusion matrix CSV
    cm_csv = out_dir / "confusion_matrix.csv"
    with open(cm_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/pred"] + label_order)
        for i, label in enumerate(label_order):
            writer.writerow([label] + [f"{cm[i, j]:.6f}" for j in range(len(label_order))])

    # Save confusion matrix figure
    cm_png = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, label_order, cm_png)

    # Save per-window predictions
    pred_csv = out_dir / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "true_label", "pred_label", "pred_conf"],
        )
        writer.writeheader()
        for i in range(len(y_pred)):
            pred_idx = int(y_pred[i])
            pred_label = label_order[pred_idx]
            pred_conf = float(probs[i, pred_idx])
            writer.writerow(
                {
                    "index": i,
                    "true_label": y_true_labels[i],
                    "pred_label": pred_label,
                    "pred_conf": f"{pred_conf:.6f}",
                }
            )

    summary = {
        "version": "beamsense_v2_states",
        "test_csv": str(test_csv),
        "model_path": str(model_path),
        "label_map_json": str(label_map_json),
        "out_dir": str(out_dir),
        "loss": float(loss),
        "accuracy": float(acc),
        "label_order": label_order,
        "artifacts": {
            "confusion_matrix_csv": str(cm_csv),
            "confusion_matrix_png": str(cm_png),
            "predictions_csv": str(pred_csv),
        },
        "classification_report": report,
    }

    summary_json = out_dir / "eval_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] test_csv:", test_csv)
    print("[INFO] model_path:", model_path)
    print("[INFO] loss:", loss)
    print("[INFO] accuracy:", acc)
    print("[INFO] confusion_matrix_csv:", cm_csv)
    print("[INFO] confusion_matrix_png:", cm_png)
    print("[INFO] predictions_csv:", pred_csv)
    print("[INFO] summary_json:", summary_json)


if __name__ == "__main__":
    main()