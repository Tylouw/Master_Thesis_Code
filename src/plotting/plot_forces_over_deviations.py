#!/usr/bin/env python3
"""
Plot tcp_force_x/y/z from multiple CSV+JSON (metadata) file pairs.

Assumptions:
- Each recording consists of two files with same basename:
    recording_01.csv
    recording_01.json
- JSON contains key: "deviation": [dx, dy, dz, dRx, dRy, dRz]
  where dx is in meters (e.g., 0.0002 -> 0.2 mm) as in your example.
- CSV contains columns:
    "tcp_force_x[N]", "tcp_force_y[N]", "tcp_force_z[N]"
  Optionally a time column ("time", "timestamp", etc.) is used if found;
  otherwise sample index is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


FORCE_COLS = ["tcp_force_x[N]", "tcp_force_y[N]", "tcp_force_z[N]"]


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: try common time column names; return None if not found."""
    candidates = [
        "time",
        "Time",
        "t",
        "timestamp",
        "Timestamp",
        "time_s",
        "time[ s ]",
        "time[s]",
        "time_sec",
        "secs",
        "sec",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_metadata(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_deviation_label(deviation: List[float]) -> str:
    """
    Default label: dx in mm (because your use-case focuses on initial x deviation).
    Example: dx=0.0002 -> 'dx=0.200 mm'
    """
    if not deviation or len(deviation) < 1:
        return "dx=?"
    dx_mm = deviation[0] * 1000.0  # meters -> mm
    return f"dx={dx_mm:.3f} mm"


def discover_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    """
    Return list of (csv_path, json_path) pairs based on matching basenames.
    """
    csvs = {p.stem: p for p in folder.glob("*.csv")}
    jsons = {p.stem: p for p in folder.glob("*.json")}

    common = sorted(set(csvs).intersection(jsons))
    pairs = [(csvs[name], jsons[name]) for name in common]

    if not pairs:
        raise FileNotFoundError(
            f"No matching CSV+JSON pairs found in: {folder}\n"
            f"Need files like 'recording.csv' and 'recording.json' with same basename."
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TCP forces from CSV+JSON pairs.")
    parser.add_argument("folder", type=str, help="Folder containing CSV+JSON pairs")
    parser.add_argument(
        "--sort-by",
        choices=["dx", "filename"],
        default="dx",
        help="Order curves by x deviation (default) or by filename",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="If set, save figure to this path (e.g. forces.png) instead of only showing.",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    pairs = discover_pairs(folder)

    # Load all recordings
    recordings = []
    for csv_path, json_path in pairs:
        df = pd.read_csv(csv_path)

        missing = [c for c in FORCE_COLS if c not in df.columns]
        if missing:
            raise KeyError(
                f"CSV '{csv_path.name}' is missing columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        meta = load_metadata(json_path)
        deviation = meta.get("deviation", [])
        label = format_deviation_label(deviation)

        time_col = find_time_column(df)
        if time_col is not None:
            x = df[time_col].to_numpy()
            x_label = time_col
        else:
            x = df.index.to_numpy()
            x_label = "sample index"

        recordings.append(
            {
                "name": csv_path.stem,
                "csv": csv_path,
                "json": json_path,
                "df": df,
                "x": x,
                "x_label": x_label,
                "deviation": deviation,
                "label": label,
            }
        )

    # Sorting
    if args.sort_by == "dx":
        def dx_val(r) -> float:
            dev = r["deviation"]
            return float(dev[0]) if isinstance(dev, list) and len(dev) > 0 else float("inf")
        recordings.sort(key=dx_val)
    else:
        recordings.sort(key=lambda r: r["name"])

    # Plot
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
    titles = ["TCP Force X", "TCP Force Y", "TCP Force Z"]
    ylabels = ["Force X [N]", "Force Y [N]", "Force Z [N]"]

    for ax, col, title, ylabel in zip(axes, FORCE_COLS, titles, ylabels):
        for r in recordings:
            ax.plot(r["x"], r["df"][col].to_numpy(), label=r["label"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axes[-1].set_xlabel(recordings[0]["x_label"])
    axes[0].legend(loc="best", title="Deviation", ncol=2)

    fig.suptitle("TCP Forces for Different Initial Deviations", y=0.98)
    fig.tight_layout()

    if args.save:
        out = Path(args.save).expanduser().resolve()
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
