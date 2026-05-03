#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


PEER_IDS = ["rtax52_a", "rtax52_b", "rtax52_c"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def find_peer_valid_col(df: pd.DataFrame, peer_id: str) -> Optional[str]:
    suffix = peer_id.split("_")[-1]  # a / b / c
    candidates = [
        f"{peer_id}_valid_frames",
        f"{peer_id}_count",
        f"{peer_id}_frames",
        f"{suffix}_valid_frames",
        f"peer_{suffix}_valid_frames",
        f"{peer_id}_n",
    ]
    return pick_col(df, candidates)


def load_segments(seg_path: Path) -> pd.DataFrame:
    seg = pd.read_csv(seg_path)
    seg["start_sec"] = pd.to_numeric(seg["start_sec"], errors="coerce")
    seg["end_sec"] = pd.to_numeric(seg["end_sec"], errors="coerce")
    seg = seg.dropna(subset=["start_sec", "end_sec"]).sort_values("start_sec").reset_index(drop=True)
    return seg


def assign_segment_info(t: float, seg_df: pd.DataFrame) -> Dict[str, Any]:
    hit = seg_df[(seg_df["start_sec"] <= t) & (t < seg_df["end_sec"])]
    if len(hit) == 0:
        return {
            "segment_id": None,
            "state_label": None,
            "segment_start_sec": None,
            "segment_end_sec": None,
            "is_outside_segment": 1,
        }
    row = hit.iloc[0]
    return {
        "segment_id": row.get("segment_id", None),
        "state_label": row.get("state_label", None),
        "segment_start_sec": float(row["start_sec"]),
        "segment_end_sec": float(row["end_sec"]),
        "is_outside_segment": 0,
    }


def compute_dominant_peer(row: pd.Series, peer_valid_cols: Dict[str, Optional[str]]) -> tuple[Optional[str], float]:
    vals = {}
    total = 0.0
    for pid, col in peer_valid_cols.items():
        v = 0.0
        if col is not None and col in row:
            try:
                v = float(row[col])
            except Exception:
                v = 0.0
        vals[pid] = v
        total += v

    if total <= 0:
        return None, 0.0

    dominant_peer = max(vals, key=vals.get)
    dominant_ratio = vals[dominant_peer] / total if total > 0 else 0.0
    return dominant_peer, dominant_ratio


def resolve_time_axis(df: pd.DataFrame, window_interval: float) -> tuple[pd.DataFrame, str]:
    """
    Resolve the time axis used for assigning each fused window to a labelled segment.

    The function first looks for an explicit time column. If no explicit time
    column is available, it derives the window centre as:
        time_center_sec = (window_idx - 0.5) * window_interval

    This convention matches the state_test_v2.csv time-centre convention used
    in the project pipeline.
    """
    time_col = pick_col(df, [
        "time_center_sec",
        "window_center_sec",
        "center_sec",
        "t_sec",
        "time_sec",
        "window_start_sec",
        "start_sec",
    ])

    if time_col is not None:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        df["time_center_sec"] = df[time_col].astype(float)
        return df, f"explicit:{time_col}"

    idx_col = pick_col(df, [
        "window_idx",
        "window_index",
        "idx",
        "index",
        "row_index",
    ])
    if idx_col is not None:
        df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce")
        df = df.dropna(subset=[idx_col]).sort_values(idx_col).reset_index(drop=True)
        df["time_center_sec"] = (df[idx_col].astype(float) - 0.5) * float(window_interval)
        return df, f"derived_from:{idx_col}"

    raise RuntimeError(
        f"Cannot find time column or window index column. Columns={list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser(description="Build v3 state index with quality metadata.")
    ap.add_argument("--sessions-root", required=True)
    ap.add_argument("--segments-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--window-interval", type=float, default=0.1)
    ap.add_argument("--min-valid-peer-count", type=int, default=1)
    ap.add_argument("--min-total-valid-frames", type=int, default=0)
    ap.add_argument("--boundary-margin-sec", type=float, default=0.5)
    ap.add_argument("--include-sessions", type=str, default="")
    args = ap.parse_args()

    sessions_root = Path(args.sessions_root)
    segments_root = Path(args.segments_root)
    out_csv = Path(args.out_csv)

    ensure_parent(out_csv)

    include_sessions = None
    if args.include_sessions.strip():
        include_sessions = {x.strip() for x in args.include_sessions.split(",") if x.strip()}

    session_dirs = sorted([p for p in sessions_root.iterdir() if p.is_dir()])
    rows: List[Dict[str, Any]] = []

    summary = {
        "version": "beamsense_v3_index",
        "sessions_root": str(sessions_root),
        "segments_root": str(segments_root),
        "sessions_used": [],
        "num_sessions_used": 0,
        "window_interval_sec": args.window_interval,
        "min_valid_peer_count": args.min_valid_peer_count,
        "min_total_valid_frames": args.min_total_valid_frames,
        "boundary_margin_sec": args.boundary_margin_sec,
        "filter_stats": {
            "drop_boundary_margin": 0,
            "drop_outside_segment": 0,
            "drop_low_valid_peer_count": 0,
            "drop_low_total_valid_frames": 0,
            "kept_for_training": 0,
        },
        "per_session_debug": {},
    }

    for session_dir in session_dirs:
        session_name = session_dir.name
        if include_sessions is not None and session_name not in include_sessions:
            continue

        fused_dir = session_dir / "fused"
        fusion_index_csv = fused_dir / "fusion_index.csv"
        seg_csv = segments_root / f"{session_name}_segments.csv"

        if not fusion_index_csv.exists():
            print(f"[WARN] skip {session_name}: missing {fusion_index_csv}")
            continue
        if not seg_csv.exists():
            print(f"[WARN] skip {session_name}: missing {seg_csv}")
            continue

        df = load_csv(fusion_index_csv)
        seg = load_segments(seg_csv)

        df, time_source = resolve_time_axis(df, args.window_interval)

        peer_valid_cols = {pid: find_peer_valid_col(df, pid) for pid in PEER_IDS}

        session_debug = {
            "windows_seen": int(len(df)),
            "time_source": time_source,
            "drop_boundary_margin": 0,
            "drop_outside_segment": 0,
            "drop_low_valid_peer_count": 0,
            "drop_low_total_valid_frames": 0,
            "kept_for_training": 0,
        }

        for i, row in df.iterrows():
            t = float(row["time_center_sec"])

            seg_info = assign_segment_info(t, seg)

            peer_vals = {}
            valid_peer_count = 0
            total_valid_frames = 0.0
            for pid, col in peer_valid_cols.items():
                v = 0.0
                if col is not None and col in row:
                    try:
                        v = float(row[col])
                    except Exception:
                        v = 0.0
                peer_vals[f"{pid}_valid_frames"] = v
                total_valid_frames += v
                if v > 0:
                    valid_peer_count += 1

            dominant_peer, dominant_peer_ratio = compute_dominant_peer(row, peer_valid_cols)

            is_boundary = 0
            if seg_info["is_outside_segment"] == 0:
                seg_start = float(seg_info["segment_start_sec"])
                seg_end = float(seg_info["segment_end_sec"])
                if (t - seg_start) < args.boundary_margin_sec or (seg_end - t) < args.boundary_margin_sec:
                    is_boundary = 1

            is_low_valid_peer = 1 if valid_peer_count < args.min_valid_peer_count else 0
            is_low_total_valid = 1 if total_valid_frames < args.min_total_valid_frames else 0

            keep_for_training = 1
            if seg_info["is_outside_segment"] == 1:
                keep_for_training = 0
                session_debug["drop_outside_segment"] += 1
            elif is_boundary == 1:
                keep_for_training = 0
                session_debug["drop_boundary_margin"] += 1
            elif is_low_valid_peer == 1:
                keep_for_training = 0
                session_debug["drop_low_valid_peer_count"] += 1
            elif is_low_total_valid == 1:
                keep_for_training = 0
                session_debug["drop_low_total_valid_frames"] += 1
            else:
                session_debug["kept_for_training"] += 1

            out_row = {
                "session_name": session_name,
                "row_index": int(i),
                "time_center_sec": t,
                "segment_id": seg_info["segment_id"],
                "state_label": seg_info["state_label"],
                "segment_start_sec": seg_info["segment_start_sec"],
                "segment_end_sec": seg_info["segment_end_sec"],
                "valid_peer_count": int(valid_peer_count),
                "total_valid_frames": float(total_valid_frames),
                "dominant_peer": dominant_peer,
                "dominant_peer_ratio": float(dominant_peer_ratio),
                "is_outside_segment": int(seg_info["is_outside_segment"]),
                "is_boundary": int(is_boundary),
                "is_low_valid_peer_count": int(is_low_valid_peer),
                "is_low_total_valid_frames": int(is_low_total_valid),
                "keep_for_training": int(keep_for_training),
                "time_source": time_source,
            }
            out_row.update(peer_vals)
            rows.append(out_row)

        summary["sessions_used"].append(session_name)
        summary["per_session_debug"][session_name] = session_debug
        for k in summary["filter_stats"].keys():
            summary["filter_stats"][k] += session_debug[k]

    summary["num_sessions_used"] = len(summary["sessions_used"])

    if len(rows) == 0:
        raise RuntimeError("No rows collected. Check sessions-root / segments-root / include-sessions.")

    full_df = pd.DataFrame(rows)
    full_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    trainable_csv = out_csv.with_name(out_csv.stem + "_trainable.csv")
    train_df = full_df[full_df["keep_for_training"] == 1].copy()
    train_df.to_csv(trainable_csv, index=False, encoding="utf-8-sig")

    state_counts = train_df["state_label"].value_counts(dropna=True).to_dict()
    summary["num_rows_total"] = int(len(full_df))
    summary["num_rows_trainable"] = int(len(train_df))
    summary["state_counts_trainable"] = state_counts

    summary_json = out_csv.with_name(out_csv.stem + ".summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote full index     : {out_csv}")
    print(f"[OK] wrote trainable csv : {trainable_csv}")
    print(f"[OK] wrote summary json  : {summary_json}")


if __name__ == "__main__":
    main()