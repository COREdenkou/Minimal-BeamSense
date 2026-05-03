#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def required_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}. Columns={list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(
        description="Merge predictions_v2.csv with state_test_v2.csv and state_index_v3 metadata."
    )
    ap.add_argument("--pred-csv", required=True, help="eval/predictions_v2.csv")
    ap.add_argument("--state-test-csv", required=True, help="split/state_test_v2.csv")
    ap.add_argument("--state-index-v3-csv", required=True, help="state_index_v3_trainable.csv")
    ap.add_argument("--out-csv", required=True, help="output merged csv")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    state_test_csv = Path(args.state_test_csv)
    state_index_csv = Path(args.state_index_v3_csv)
    out_csv = Path(args.out_csv)

    if not pred_csv.exists():
        raise RuntimeError(f"pred_csv not found: {pred_csv}")
    if not state_test_csv.exists():
        raise RuntimeError(f"state_test_csv not found: {state_test_csv}")
    if not state_index_csv.exists():
        raise RuntimeError(f"state_index_v3_csv not found: {state_index_csv}")

    ensure_parent(out_csv)

    pred = pd.read_csv(pred_csv)
    stest = pd.read_csv(state_test_csv)
    sidx = pd.read_csv(state_index_csv)

    required_cols(pred, ["index", "true_label", "pred_label"], "predictions_v2.csv")
    required_cols(stest, ["session_name", "state_label", "time_center_sec"], "state_test_v2.csv")
    required_cols(
        sidx,
        [
            "session_name",
            "time_center_sec",
            "state_label",
            "valid_peer_count",
            "total_valid_frames",
            "keep_for_training",
        ],
        "state_index_v3_trainable.csv",
    )

    # 1) Align predictions with state_test rows by row index
    stest = stest.reset_index().rename(columns={"index": "row_index"})
    pred = pred.rename(columns={"index": "row_index"})

    merged = stest.merge(
        pred,
        on="row_index",
        how="inner",
        suffixes=("_stest", "_pred"),
    )

    merged["gt_label"] = merged["state_label"].astype(str)
    merged["pred_label"] = merged["pred_label"].astype(str)
    merged["is_correct"] = (merged["gt_label"] == merged["pred_label"]).astype(int)

    # 2) Merge v3 quality metadata
    meta_cols = [
        "session_name",
        "time_center_sec",
        "state_label",
        "segment_id",
        "segment_start_sec",
        "segment_end_sec",
        "valid_peer_count",
        "total_valid_frames",
        "dominant_peer",
        "dominant_peer_ratio",
        "is_outside_segment",
        "is_boundary",
        "is_low_valid_peer_count",
        "is_low_total_valid_frames",
        "keep_for_training",
        "rtax52_a_valid_frames",
        "rtax52_b_valid_frames",
        "rtax52_c_valid_frames",
        "time_source",
    ]
    meta_cols = [c for c in meta_cols if c in sidx.columns]
    meta = sidx[meta_cols].copy()

    merged["time_center_sec_round"] = pd.to_numeric(merged["time_center_sec"], errors="coerce").round(4)
    meta["time_center_sec_round"] = pd.to_numeric(meta["time_center_sec"], errors="coerce").round(4)

    merged = merged.merge(
        meta,
        left_on=["session_name", "time_center_sec_round", "state_label"],
        right_on=["session_name", "time_center_sec_round", "state_label"],
        how="left",
        suffixes=("_stest", "_v3"),
    )

    # Remove temporary helper columns
    merged = merged.drop(columns=["time_center_sec_round"], errors="ignore")

    # Estimate the metadata match rate
    if "valid_peer_count_v3" in merged.columns:
        metadata_match_rate = float(merged["valid_peer_count_v3"].notna().mean())
    elif "dominant_peer" in merged.columns:
        metadata_match_rate = float(merged["dominant_peer"].notna().mean())
    else:
        metadata_match_rate = 0.0

    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote merged csv: {out_csv}")
    print(f"[INFO] rows: {len(merged)}")
    print(f"[INFO] metadata_match_rate: {metadata_match_rate:.4f}")


if __name__ == "__main__":
    main()