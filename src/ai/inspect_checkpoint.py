#!/usr/bin/env python3
"""
inspect_checkpoint.py

Usage:
  python inspect_checkpoint.py path/to/checkpoint.pt

Prints:
- checkpoint keys
- epoch / val metrics if present
- model config (dataclass or dict)
- parameter count (if we can reconstruct the model)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass-like or namespace-like objects to a plain dict when possible."""
    # dict already
    if isinstance(obj, dict):
        return obj
    # dataclass
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
    except Exception:
        pass
    # objects with __dict__
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    # fallback
    return {"_repr": repr(obj)}


def _print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_path", type=str, help="Path to .pt checkpoint")
    ap.add_argument("--no-model", action="store_true", help="Do not try to instantiate the model")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    _print_section("Checkpoint file")
    print("path:", str(ckpt_path))
    print("size:", ckpt_path.stat().st_size, "bytes")
    print("type:", type(ckpt).__name__)

    if isinstance(ckpt, dict):
        _print_section("Top-level keys")
        for k in sorted(ckpt.keys()):
            v = ckpt[k]
            t = type(v).__name__
            if isinstance(v, (str, int, float, bool, type(None))):
                print(f"- {k}: ({t}) {v}")
            elif isinstance(v, (list, tuple)) and len(v) <= 20:
                print(f"- {k}: ({t}, len={len(v)}) {v}")
            elif isinstance(v, dict):
                print(f"- {k}: ({t}, keys={len(v)})")
            else:
                print(f"- {k}: ({t})")

        _print_section("Common metadata (if present)")
        for key in ["epoch", "val_acc", "val_loss", "best_val_acc", "best_val_bal_acc", "val_groups", "seed"]:
            if key in ckpt:
                print(f"{key}: {ckpt[key]}")
        def format_hms(seconds: float) -> str:
            s = int(round(seconds))
            h = s // 3600
            m = (s % 3600) // 60
            sec = s % 60
            return f"{h}:{m:02d}:{sec:02d}"

        for key in ["time_to_best_s", "total_train_time_s"]:
            if key in ckpt:
                val = float(ckpt[key])
                print(f"{key}: {val:.2f}  ({format_hms(val)})")


        best_stats = ckpt.get("best_stats")
        if isinstance(best_stats, dict):
            _print_section("Best stats (from evaluate)")
            # Stable ordering
            for k in [
                "selection_metric", "best_threshold",
                "tp", "fp", "tn", "fn",
                "acc", "bal_acc", "precision", "recall", "f1", "specificity",
            ]:
                if k in best_stats:
                    print(f"{k}: {best_stats[k]}")
        else:
            print("\nNo 'best_stats' found in checkpoint.")

        # Model config
        _print_section("Model config")
        model_cfg = ckpt.get("model_cfg", None)
        if model_cfg is None:
            print("No 'model_cfg' field found in checkpoint.")
        else:
            cfg_dict = _to_plain_dict(model_cfg)
            # Pretty-print in stable order
            for k in sorted(cfg_dict.keys()):
                print(f"{k}: {cfg_dict[k]}")

        # If requested, try reconstructing model to show parameter count
        if args.no_model:
            return 0

        if model_cfg is None:
            print("\nSkipping model reconstruction (no model_cfg).")
            return 0

        # Try importing your model code
        try:
            # If script is in src/, this will work if you run from project root.
            from patchtst import PatchTSTClassifier, PatchTSTConfig  # type: ignore
        except Exception as e:
            print("\nCould not import patchtst.PatchTSTClassifier/PatchTSTConfig.")
            print("Run from project root or adjust PYTHONPATH.")
            print("Error:", repr(e))
            return 0

        # Rebuild PatchTSTConfig
        cfg_dict = _to_plain_dict(model_cfg)
        try:
            cfg_obj = PatchTSTConfig(**cfg_dict)  # if dict saved
        except Exception:
            # maybe model_cfg in checkpoint already a PatchTSTConfig instance
            cfg_obj = model_cfg

        try:
            model = PatchTSTClassifier(cfg_obj)
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _print_section("Model reconstruction")
            print("Model class:", type(model).__name__)
            print("Total params:", n_params)
            print("Trainable params:", n_trainable)

            # Optional: try loading weights to confirm compatibility
            state = ckpt.get("model_state", None)
            if isinstance(state, dict):
                missing, unexpected = model.load_state_dict(state, strict=False)
                print("Loaded model_state with strict=False")
                print("Missing keys:", len(missing))
                print("Unexpected keys:", len(unexpected))
            else:
                print("No 'model_state' dict found (or wrong type).")

        except Exception as e:
            print("\nModel reconstruction failed.")
            print("Error:", repr(e))

    else:
        _print_section("Checkpoint is not a dict")
        print(repr(ckpt))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
