#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run blind inference for a Minimal BeamSense session.

The script loads a trained Keras model, predicts one class per inference window,
and then applies fixed-length majority voting to produce interval-level results.
All paths are provided by command-line arguments so that the script can be reused
on different machines.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras


def load_label_order(label_map_json: Path):
    """Read class order from a label-map JSON file."""
    with open(label_map_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if isinstance(obj, dict):
        if "label_order" in obj:
            return [str(x) for x in obj["label_order"]]

        if "index_to_label" in obj and isinstance(obj["index_to_label"], dict):
            items = sorted(obj["index_to_label"].items(), key=lambda kv: int(kv[0]))
            return [str(v) for _, v in items]

        if "label_to_index" in obj and isinstance(obj["label_to_index"], dict):
            items = sorted(obj["label_to_index"].items(), key=lambda kv: int(kv[1]))
            return [str(k) for k, _ in items]

        # Some label maps are saved directly as {label: index}.
        if all(isinstance(v, int) for v in obj.values()):
            items = sorted(obj.items(), key=lambda kv: int(kv[1]))
            return [str(k) for k, _ in items]

    raise RuntimeError(f"Cannot parse label order from {label_map_json}")


def find_col(cols, candidates):
    """Find the first matching column name, case-insensitively."""
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def make_time_columns(df: pd.DataFrame, interval_sec: float):
    """Build start, end and centre timestamps for each inference window."""
    cols = list(df.columns)

    start_col = find_col(cols, [
        "start_sec", "window_start_sec", "time_start_sec",
        "t_start", "start", "begin_sec",
    ])
    end_col = find_col(cols, [
        "end_sec", "window_end_sec", "time_end_sec",
        "t_end", "end", "finish_sec",
    ])
    center_col = find_col(cols, [
        "center_sec", "window_center_sec", "time_center_sec",
        "t_center", "time_sec",
    ])

    out = pd.DataFrame()
    n = len(df)

    if start_col and end_col:
        out["start_sec"] = pd.to_numeric(df[start_col], errors="coerce")
        out["end_sec"] = pd.to_numeric(df[end_col], errors="coerce")
        out["center_sec"] = (out["start_sec"] + out["end_sec"]) / 2.0
    elif center_col:
        out["center_sec"] = pd.to_numeric(df[center_col], errors="coerce")
        out["start_sec"] = out["center_sec"] - interval_sec / 2.0
        out["end_sec"] = out["center_sec"] + interval_sec / 2.0
    else:
        out["start_sec"] = np.arange(n) * interval_sec
        out["end_sec"] = out["start_sec"] + interval_sec
        out["center_sec"] = out["start_sec"] + interval_sec / 2.0

    # Use an index-derived time axis if any row has missing time metadata.
    fallback_start = np.arange(n) * interval_sec
    fallback_end = fallback_start + interval_sec
    fallback_center = fallback_start + interval_sec / 2.0

    out["start_sec"] = out["start_sec"].fillna(pd.Series(fallback_start))
    out["end_sec"] = out["end_sec"].fillna(pd.Series(fallback_end))
    out["center_sec"] = out["center_sec"].fillna(pd.Series(fallback_center))

    return out


def vote_intervals(sample_df: pd.DataFrame, label_order, vote_sec: float):
    """Convert sample-level predictions into interval-level majority decisions."""
    df = sample_df.copy()
    df["interval_index"] = np.floor(df["center_sec"] / vote_sec).astype(int)
    df["interval_start_sec"] = df["interval_index"] * vote_sec
    df["interval_end_sec"] = df["interval_start_sec"] + vote_sec

    rows = []
    prob_cols = [f"prob_{x}" for x in label_order if f"prob_{x}" in df.columns]

    for idx, g in df.groupby("interval_index", sort=True):
        counts = g["pred_label"].value_counts()
        majority_label = str(counts.index[0])
        majority_count = int(counts.iloc[0])
        n = int(len(g))
        majority_ratio = majority_count / n if n else 0.0

        row = {
            "interval_index": int(idx),
            "start_sec": float(g["interval_start_sec"].iloc[0]),
            "end_sec": float(g["interval_end_sec"].iloc[0]),
            "n_samples": n,
            "majority_label": majority_label,
            "majority_class_id": int(label_order.index(majority_label)) if majority_label in label_order else -1,
            "majority_count": majority_count,
            "majority_ratio": float(majority_ratio),
            "mean_pred_conf": float(g["pred_conf"].mean()) if n else 0.0,
        }

        for pc in prob_cols:
            row[f"mean_{pc}"] = float(g[pc].mean()) if n else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Run Keras inference and majority voting for one session.")
    ap.add_argument("--infer-csv", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--label-map-json", required=True)
    ap.add_argument("--data-generator-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--session-name", required=True)
    ap.add_argument("--vote-sec", type=float, default=5.0)
    ap.add_argument("--window-interval", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    infer_csv = Path(args.infer_csv)
    model_path = Path(args.model_path)
    label_map_json = Path(args.label_map_json)
    data_generator_dir = Path(args.data_generator_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not infer_csv.is_file():
        raise FileNotFoundError(infer_csv)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not label_map_json.is_file():
        raise FileNotFoundError(label_map_json)

    sys.path.insert(0, str(data_generator_dir))
    from dataGenerator_states import DataGenerator  # noqa: E402

    label_order = load_label_order(label_map_json)

    print("[INFO] label_order:", label_order)
    print("[INFO] loading model:", model_path)

    model = keras.models.load_model(model_path)

    gen = DataGenerator(
        dataset_csv=str(infer_csv),
        batchsize=args.batch_size,
        shuffle=False,
        to_categorical=True,
        label_order=label_order,
        normalize_by_180=True,
    )

    probs = model.predict(gen, verbose=1)
    probs = np.asarray(probs)

    src_df = pd.read_csv(infer_csv)
    n_pred = probs.shape[0]

    if n_pred > len(src_df):
        raise RuntimeError(f"Prediction count {n_pred} > input rows {len(src_df)}")

    if n_pred < len(src_df):
        print(f"[WARN] Prediction count {n_pred} < input rows {len(src_df)}. Truncating input rows.")
        src_df = src_df.iloc[:n_pred].reset_index(drop=True)
    else:
        src_df = src_df.reset_index(drop=True)

    time_df = make_time_columns(src_df, interval_sec=args.window_interval).iloc[:n_pred].reset_index(drop=True)

    pred_ids = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    pred_labels = [label_order[i] for i in pred_ids]

    out = pd.DataFrame({
        "sample_index": np.arange(n_pred),
        "session_name": args.session_name,
        "start_sec": time_df["start_sec"].astype(float),
        "end_sec": time_df["end_sec"].astype(float),
        "center_sec": time_df["center_sec"].astype(float),
        "pred_label": pred_labels,
        "pred_class_id": pred_ids.astype(int),
        "pred_conf": pred_conf.astype(float),
    })

    for i, lab in enumerate(label_order):
        out[f"prob_{lab}"] = probs[:, i].astype(float)

    sample_csv = out_dir / "sample_predictions.csv"
    out.to_csv(sample_csv, index=False, encoding="utf-8-sig")

    interval_df = vote_intervals(out, label_order=label_order, vote_sec=args.vote_sec)
    interval_csv = out_dir / f"interval_predictions_{int(args.vote_sec)}s.csv"
    interval_df.to_csv(interval_csv, index=False, encoding="utf-8-sig")

    summary = {
        "session_name": args.session_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "infer_csv": str(infer_csv),
        "model_path": str(model_path),
        "label_map_json": str(label_map_json),
        "label_order": label_order,
        "vote_sec": args.vote_sec,
        "window_interval": args.window_interval,
        "num_samples": int(n_pred),
        "num_intervals": int(len(interval_df)),
        "sample_predictions_csv": str(sample_csv),
        "interval_predictions_csv": str(interval_csv),
    }

    if len(interval_df):
        overall_counts = out["pred_label"].value_counts()
        summary["overall_majority_label"] = str(overall_counts.index[0])
        summary["overall_majority_ratio"] = float(overall_counts.iloc[0] / len(out))

    summary_json = out_dir / "inference_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] wrote:", sample_csv)
    print("[OK] wrote:", interval_csv)
    print("[OK] wrote:", summary_json)


if __name__ == "__main__":
    main()
