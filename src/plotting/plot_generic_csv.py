#!/usr/bin/env python3
"""
plot_csv_stacked.py

Example:
  python plot_csv_stacked.py data.csv tcp_force_x[N] tcp_force_y[N] tcp_force_z[N]
  python src/plot_generic_csv.py ForceDataNovo/Old_Fixture/2025_08_29/2025_08_29_15_15_13_True.csv FX_Base FY_Base FZ_Base

With explicit x column:
  python plot_csv_stacked.py data.csv --xcol time[s] tcp_force_x[N] tcp_force_y[N]

Save to file:
  python plot_csv_stacked.py data.csv --xcol time --save plot.png col1 col2 col3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot one or more CSV columns as stacked subplots."
    )
    p.add_argument("csv", type=Path, help="Path to CSV file")
    p.add_argument(
        "columns",
        nargs="+",
        help="One or more column headers to plot (each becomes its own subplot)",
    )
    p.add_argument(
        "--xcol",
        default=None,
        help="Optional x-axis column header. If omitted, the row index is used.",
    )
    p.add_argument(
        "--delimiter",
        default=None,
        help="Optional delimiter override (e.g. ',' or ';'). If omitted, pandas auto-detects reasonably well.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    p.add_argument(
        "--grid",
        action="store_true",
        help="Show grid on all subplots.",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure (e.g. plot.png). If omitted, opens an interactive window.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV file not found: {args.csv}", file=sys.stderr)
        return 2

    # Read CSV
    try:
        df = pd.read_csv(args.csv, sep=args.delimiter) if args.delimiter else pd.read_csv(args.csv)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}", file=sys.stderr)
        return 2

    # Validate columns
    missing = [c for c in ([args.xcol] if args.xcol else []) + args.columns if c not in df.columns]
    if missing:
        print("ERROR: Missing column(s):", ", ".join(missing), file=sys.stderr)
        print("Available columns:", ", ".join(map(str, df.columns)), file=sys.stderr)
        return 2

    x = df[args.xcol] if args.xcol else df.index

    n = len(args.columns)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, max(2.2 * n, 3.5)))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, args.columns):
        ax.plot(x, df[col])
        ax.set_ylabel(col)
        if args.grid:
            ax.grid(True)

    axes[-1].set_xlabel(args.xcol if args.xcol else "index")

    if args.title:
        fig.suptitle(args.title)

    fig.tight_layout()

    if args.save:
        try:
            fig.savefig(args.save, dpi=200, bbox_inches="tight")
            print(f"Saved plot to: {args.save}")
        except Exception as e:
            print(f"ERROR: Failed to save figure: {e}", file=sys.stderr)
            return 2
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
