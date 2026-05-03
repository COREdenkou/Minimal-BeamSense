#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a simple segment file for blind-inference window construction.

The generated label is a placeholder used to create windows for an unlabelled
session. It should not be interpreted as ground truth for evaluation.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def main():
    ap = argparse.ArgumentParser(description="Create a placeholder segment CSV for inference.")
    ap.add_argument("--session-name", required=True)
    ap.add_argument("--segments-root", required=True)
    ap.add_argument("--duration-sec", type=float, required=True)
    ap.add_argument("--label", default="empty")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--template-csv", default=None)
    args = ap.parse_args()

    segments_root = Path(args.segments_root)
    segments_root.mkdir(parents=True, exist_ok=True)

    out_csv = segments_root / f"{args.session_name}_segments.csv"

    start = float(args.start_sec)
    end = start + float(args.duration_sec)

    if args.template_csv:
        template = pd.read_csv(args.template_csv)
        cols = list(template.columns)
        row = {c: "" for c in cols}

        state_col = find_col(cols, ["state_label", "label", "state", "class"])
        start_col = find_col(cols, ["start_sec", "start", "t_start", "begin_sec", "begin"])
        end_col = find_col(cols, ["end_sec", "end", "t_end", "finish_sec", "finish"])
        session_col = find_col(cols, ["session_name", "session", "sample_name"])

        if state_col is None or start_col is None or end_col is None:
            raise RuntimeError(
                f"Cannot infer required columns from template. Columns={cols}"
            )

        row[state_col] = args.label
        row[start_col] = start
        row[end_col] = end
        if session_col:
            row[session_col] = args.session_name

        df = pd.DataFrame([row], columns=cols)
    else:
        df = pd.DataFrame([{
            "session_name": args.session_name,
            "state_label": args.label,
            "start_sec": start,
            "end_sec": end,
        }])

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "session_name": args.session_name,
        "segments_csv": str(out_csv),
        "label": args.label,
        "start_sec": start,
        "end_sec": end,
        "duration_sec": args.duration_sec,
        "note": "Placeholder segment for inference window construction. Label is not used as ground truth.",
    }
    meta_path = segments_root / f"{args.session_name}_segments_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] wrote:", out_csv)
    print("[OK] wrote:", meta_path)


if __name__ == "__main__":
    main()
