#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Tuple, List

import pandas as pd


QUALITY_COLS = [
    "valid_peer_count",
    "total_valid_frames",
    "rtax52_a_valid_frames",
    "rtax52_b_valid_frames",
    "rtax52_c_valid_frames",
    "dominant_peer",
    "dominant_peer_ratio",
    "keep_for_training",
    "segment_id",
    "segment_start_sec",
    "segment_end_sec",
    "time_source",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"CSV not found: {path}")
    return pd.read_csv(path)


def choose_merge_strategy(base_df: pd.DataFrame, v3_df: pd.DataFrame) -> Tuple[str, List[str]]:
    # Prefer row_index when it is available
    key1 = ["session_name", "row_index", "state_label"]
    if all(k in base_df.columns for k in key1) and all(k in v3_df.columns for k in key1):
        return "row_index", key1

    # Otherwise fall back to time_center_sec
    key2 = ["session_name", "time_center_sec", "state_label"]
    if all(k in base_df.columns for k in key2) and all(k in v3_df.columns for k in key2):
        return "time_center_sec", key2

    raise RuntimeError(
        "Cannot find compatible merge keys.\n"
        f"base columns={list(base_df.columns)}\n"
        f"v3 columns={list(v3_df.columns)}"
    )


def merge_one_split(name: str, base_csv: Path, v3_csv: Path, out_csv: Path) -> dict:
    base_df = load_csv(base_csv)
    v3_df = load_csv(v3_csv)

    strategy, keys = choose_merge_strategy(base_df, v3_df)

    # Keep only the v3 quality columns and merge keys
    use_cols = [c for c in QUALITY_COLS if c in v3_df.columns]
    for k in keys:
        if k not in use_cols:
            use_cols.insert(0, k)

    v3_small = v3_df[use_cols].copy()

    if strategy == "time_center_sec":
        base_df["time_center_sec_round"] = pd.to_numeric(base_df["time_center_sec"], errors="coerce").round(4)
        v3_small["time_center_sec_round"] = pd.to_numeric(v3_small["time_center_sec"], errors="coerce").round(4)

        left_keys = ["session_name", "time_center_sec_round", "state_label"]
        right_keys = ["session_name", "time_center_sec_round", "state_label"]

        # Avoid duplicated time_center_sec columns
        if "time_center_sec" in v3_small.columns:
            v3_small = v3_small.drop(columns=["time_center_sec"])

        merged = base_df.merge(
            v3_small,
            left_on=left_keys,
            right_on=right_keys,
            how="left",
            suffixes=("", "_v3"),
        ).drop(columns=["time_center_sec_round"], errors="ignore")
    else:
        merged = base_df.merge(
            v3_small,
            on=keys,
            how="left",
            suffixes=("", "_v3"),
        )

    # 检查匹配率
    probe_col = None
    for c in ["valid_peer_count", "dominant_peer", "total_valid_frames"]:
        if c in merged.columns and c not in base_df.columns:
            probe_col = c
            break
        if f"{c}_v3" in merged.columns:
            probe_col = f"{c}_v3"
            break

    match_rate = float(merged[probe_col].notna().mean()) if probe_col is not None else 0.0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return {
        "name": name,
        "base_csv": str(base_csv),
        "v3_csv": str(v3_csv),
        "out_csv": str(out_csv),
        "merge_strategy": strategy,
        "merge_keys": keys,
        "rows": int(len(merged)),
        "metadata_match_rate": match_rate,
        "output_columns": list(merged.columns),
    }


def main():
    ap = argparse.ArgumentParser(description="Build v3-ready training tables by merging v2 split with v3 quality metadata.")
    ap.add_argument("--base-train-csv", required=True)
    ap.add_argument("--base-val-csv", required=True)
    ap.add_argument("--base-test-csv", required=True)
    ap.add_argument("--v3-train-csv", required=True)
    ap.add_argument("--v3-val-csv", required=True)
    ap.add_argument("--v3-test-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    train_info = merge_one_split(
        "train",
        Path(args.base_train_csv),
        Path(args.v3_train_csv),
        out_dir / "state_train_v3_ready.csv",
    )
    val_info = merge_one_split(
        "val",
        Path(args.base_val_csv),
        Path(args.v3_val_csv),
        out_dir / "state_val_v3_ready.csv",
    )
    test_info = merge_one_split(
        "test",
        Path(args.base_test_csv),
        Path(args.v3_test_csv),
        out_dir / "state_test_v3_ready.csv",
    )

    summary = {
        "version": "build_training_table_v3",
        "out_dir": str(out_dir),
        "splits": {
            "train": train_info,
            "val": val_info,
            "test": test_info,
        },
    }

    summary_json = out_dir / "build_training_table_v3.summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {train_info['out_csv']}")
    print(f"[OK] wrote: {val_info['out_csv']}")
    print(f"[OK] wrote: {test_info['out_csv']}")
    print(f"[OK] wrote: {summary_json}")
    print(
        f"[INFO] match rates | "
        f"train={train_info['metadata_match_rate']:.4f}, "
        f"val={val_info['metadata_match_rate']:.4f}, "
        f"test={test_info['metadata_match_rate']:.4f}"
    )


if __name__ == "__main__":
    main()