#!/usr/bin/env python3
"""
Plot a single column from all CSV files in a directory.
Color by metadata JSON field "successful_insertion" (true/false).

Each CSV must have an associated JSON file with the same base name:
  run_001.csv  <->  run_001.json


  python src/plotting/plot_true_false_insertions.py ./test_recorded_data/real_test/ --col tcp_force_z[N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    print("ERROR: This script requires pandas. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "directory",
        type=Path,
        help="Directory containing .csv files (and matching .json metadata files).",
    )
    p.add_argument(
        "--col",
        required=True,
        help="CSV column name to plot (exact match). Example: --col 'tcp_force_x[N]'",
    )
    p.add_argument(
        "--xcol",
        default=None,
        help="Optional CSV column name for x-axis. If omitted, uses sample index.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search for CSV files recursively.",
    )
    p.add_argument(
        "--json-key",
        default="successful_insertion",
        help="Metadata JSON key that contains true/false (default: successful_insertion).",
    )
    p.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,).",
    )
    p.add_argument(
        "--success-color",
        default="tab:green",
        help="Matplotlib color for successful insertions (default: tab:green).",
    )
    p.add_argument(
        "--fail-color",
        default="tab:red",
        help="Matplotlib color for failed insertions (default: tab:red).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Line alpha (default: 0.6).",
    )
    p.add_argument(
        "--linewidth",
        type=float,
        default=1.0,
        help="Line width (default: 1.0).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, save figure to this path (e.g. out.png). If omitted, show interactively.",
    )
    return p.parse_args()


def load_success_flag(json_path: Path, key: str) -> bool | None:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

    val = meta.get(key, None)
    if isinstance(val, bool):
        return val
    return None


def main() -> int:
    args = parse_args()

    if not args.directory.exists() or not args.directory.is_dir():
        print(f"ERROR: Not a directory: {args.directory}", file=sys.stderr)
        return 2

    csv_paths = sorted(args.directory.rglob("*.csv") if args.recursive else args.directory.glob("*.csv"))
    if not csv_paths:
        print(f"No .csv files found in: {args.directory}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots()

    labeled_success = False
    labeled_fail = False
    n_plotted = 0
    n_skipped = 0

    for csv_path in csv_paths:
        json_path = csv_path.with_suffix(".json")
        success = load_success_flag(json_path, args.json_key)
        if success is None:
            print(f"Skipping (missing/invalid JSON or key): {csv_path.name}  (expected {json_path.name})", file=sys.stderr)
            n_skipped += 1
            continue

        try:
            df = pd.read_csv(csv_path, delimiter=args.delimiter)
        except Exception as e:
            print(f"Skipping (CSV read error): {csv_path.name}: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        if args.col not in df.columns:
            print(f"Skipping (missing column '{args.col}'): {csv_path.name}", file=sys.stderr)
            n_skipped += 1
            continue

        y = pd.to_numeric(df[args.col], errors="coerce").to_numpy()
        if args.xcol:
            if args.xcol not in df.columns:
                print(f"Skipping (missing x column '{args.xcol}'): {csv_path.name}", file=sys.stderr)
                n_skipped += 1
                continue
            x = pd.to_numeric(df[args.xcol], errors="coerce").to_numpy()
        else:
            x = range(len(y))

        color = args.success_color if success else args.fail_color
        label = None
        if success and not labeled_success:
            label = "successful_insertion = true"
            labeled_success = True
        elif (not success) and not labeled_fail:
            label = "successful_insertion = false"
            labeled_fail = True

        ax.plot(
            x,
            y,
            color=color,
            alpha=args.alpha,
            linewidth=args.linewidth,
            label=label,
        )
        n_plotted += 1

    if n_plotted == 0:
        print("Nothing plotted (all files skipped). See stderr messages above.", file=sys.stderr)
        return 1

    ax.set_title(f"Column: {args.col}  ({n_plotted} files plotted, {n_skipped} skipped)")
    ax.set_xlabel(args.xcol if args.xcol else "sample index")
    ax.set_ylabel(args.col)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    fig.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())