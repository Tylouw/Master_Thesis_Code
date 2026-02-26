#!/usr/bin/env python3
"""
run_learning_curve_patchtst.py

Orchestrates:
1) Build *fixed* IID benchmark splits (Group split by session/run).
2) Learning-curve experiment: train-from-scratch at multiple train fractions with repeats.
3) Optional OOD evaluation (train excluding a held-out domain value, test on that holdout).

Works with this project layout:
- src/ai/patchtst.py (model)
- src/ai/train_patchtst.py (training utilities)
- Data recorded via src/robotics/collectData.py (CSV + JSON metadata)

Run from the project root, e.g.:
  python src/ai/run_learning_curve_patchtst.py \
    --data-root test_recorded_data/real_test \
    --out-dir runs/lc_v1 \
    --model-preset small \
    --amp

Notes:
- Keeps IID val/test fixed by writing/reading out-dir/splits_iid.json
- Default features = TCP wrench 6D written by your collector
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# Local project imports (script dir is added to sys.path when executed via python src/ai/..)
from patchtst import PatchTSTClassifier, get_model_config, config_to_dict
import train_patchtst as tp


# -------------------------
# Dataset: CSV + JSON pairs
# -------------------------

DEFAULT_FEATURES_WRENCH6 = [
    "tcp_force_x[N]",
    "tcp_force_y[N]",
    "tcp_force_z[N]",
    "tcp_force_rx[Nm?]",
    "tcp_force_ry[Nm?]",
    "tcp_force_rz[Nm?]",
]


@dataclass(frozen=True)
class Sample:
    csv_path: Path
    json_path: Path
    label: float  # 1.0 success, 0.0 failure
    group: str    # session/run id for leakage-safe split
    domain: str   # e.g., object+tolerance or connector type (for stratification / OOD)
    meta: Dict


def _safe_get(meta: Dict, keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return default


def _infer_group(csv_path: Path, meta: Dict, group_key: Optional[str]) -> str:
    """
    Grouping key for Group split.

    Priority:
      1) metadata[group_key] if provided and present
      2) metadata["session_id"] if present
      3) parent folder name (common in older layouts)
      4) date prefix from meta["saved_at"] (YYYY-MM-DD)
      5) fallback: "ungrouped"
    """
    if group_key and group_key in meta:
        return str(meta[group_key])

    if "session_id" in meta:
        return str(meta["session_id"])

    parent = csv_path.parent.name
    if parent:
        return parent

    saved_at = meta.get("saved_at")
    if isinstance(saved_at, str) and len(saved_at) >= 10:
        return saved_at[:10]

    return "ungrouped"


def _infer_domain(csv_path: Path, meta: Dict, domain_key: Optional[str]) -> str:
    """
    Domain label for stratification + OOD.

    If domain_key exists in metadata, use it.
    Else, try (insertion_object, hole_size) from collectData.py.
    Else fallback to parent folder.
    """
    if domain_key and domain_key in meta:
        return str(meta[domain_key])

    obj = _safe_get(meta, ["insertion_object", "object", "obj"], default="")
    hole = _safe_get(meta, ["hole_size", "fixture", "target"], default="")
    if obj or hole:
        return "__".join([s for s in [obj, hole] if s])

    return csv_path.parent.name or "unknown_domain"


def index_samples(
    data_root: Path,
    group_key: Optional[str],
    domain_key: Optional[str],
    require_json: bool = True,
) -> List[Sample]:
    """Collect (csv,json) pairs recursively under data_root."""
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    samples: List[Sample] = []

    for csv_path in sorted(data_root.rglob("*.csv")):
        json_path = csv_path.with_suffix(".json")
        if require_json and not json_path.exists():
            continue

        meta: Dict = {}
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text())
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON: {json_path} ({e})")

        # Label: prefer collectData.py metadata
        if "successful_insertion" in meta:
            label = 1.0 if bool(meta["successful_insertion"]) else 0.0
        else:
            # Fallback: legacy filename suffix _True/_False
            try:
                label = tp.ForceCSVFolderDataset.label_from_name(csv_path.name)
            except Exception:
                continue

        group = _infer_group(csv_path, meta, group_key)
        domain = _infer_domain(csv_path, meta, domain_key)

        samples.append(
            Sample(
                csv_path=csv_path,
                json_path=json_path,
                label=label,
                group=group,
                domain=domain,
                meta=meta,
            )
        )

    if not samples:
        raise RuntimeError(f"No usable CSV/JSON samples found under: {data_root}")

    return samples


class InsertionCSVJsonDataset(Dataset):
    """
    Loads one sample per CSV file, labels from JSON metadata.

    Returns:
      x: FloatTensor [L, C]
      y: FloatTensor []  (0.0/1.0) for BCEWithLogitsLoss

    Handles variable length by truncation/padding to seq_len.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        seq_len: int,
        feature_cols: Sequence[str],
        clip_policy: str = "end",  # start|end|center
        fill_value: float = 0.0,
        normalize: str = "global",  # none|per_sample|global
        global_mean: Optional[np.ndarray] = None,
        global_std: Optional[np.ndarray] = None,
    ):
        self.samples = list(samples)
        self.seq_len = int(seq_len)
        self.feature_cols = list(feature_cols)
        self.clip_policy = clip_policy
        self.fill_value = float(fill_value)
        self.normalize = normalize

        if normalize == "global":
            if global_mean is None or global_std is None:
                raise ValueError("global normalization requires global_mean and global_std")
            self.global_mean = global_mean.astype(np.float32)
            self.global_std = global_std.astype(np.float32)
        else:
            self.global_mean = None
            self.global_std = None

    def __len__(self) -> int:
        return len(self.samples)

    def _fix_length(self, x: np.ndarray) -> np.ndarray:
        L, C = x.shape
        target = self.seq_len

        if L == target:
            return x

        if L > target:
            if self.clip_policy == "start":
                return x[:target]
            if self.clip_policy == "end":
                return x[-target:]
            if self.clip_policy == "center":
                s = (L - target) // 2
                return x[s : s + target]
            raise ValueError(f"Unknown clip_policy: {self.clip_policy}")

        # L < target -> pad
        pad = np.full((target - L, C), self.fill_value, dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        df = pd.read_csv(s.csv_path, usecols=self.feature_cols)

        x = df.to_numpy(dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self._fix_length(x)

        if self.normalize == "per_sample":
            mu = x.mean(axis=0, keepdims=True)
            sig = x.std(axis=0, keepdims=True) + 1e-6
            x = (x - mu) / sig
        elif self.normalize == "global":
            x = (x - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-6)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize}")

        x_t = torch.from_numpy(x)  # [L, C]
        y_t = torch.tensor(float(s.label), dtype=torch.float32)
        return x_t, y_t


# -------------------------
# Splitting + stratified subsampling
# -------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def group_split(
    samples: Sequence[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_groups: int = 3,
) -> Tuple[List[int], List[int], List[int], Dict]:
    """
    Split indices into train/val/test by grouping on Sample.group.

    If too few unique groups exist, falls back to a stratified random split on (domain,label).
    """
    groups = sorted({s.group for s in samples})
    by_group: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        by_group.setdefault(s.group, []).append(i)

    if len(groups) < min_groups:
        # Fallback: stratified random split on (domain,label)
        rng = _rng(seed)

        strata: Dict[Tuple[str, float], List[int]] = {}
        for i, s in enumerate(samples):
            strata.setdefault((s.domain, s.label), []).append(i)

        val_idx: List[int] = []
        test_idx: List[int] = []
        train_idx: List[int] = []

        for _key, ids0 in strata.items():
            ids = np.array(ids0, dtype=int)
            rng.shuffle(ids)
            n = len(ids)
            n_test = int(round(test_ratio * n))
            n_val = int(round(val_ratio * n))
            n_test = min(n_test, n)
            n_val = min(n_val, n - n_test)

            test_idx.extend(ids[:n_test].tolist())
            val_idx.extend(ids[n_test : n_test + n_val].tolist())
            train_idx.extend(ids[n_test + n_val :].tolist())

        return (
            sorted(train_idx),
            sorted(val_idx),
            sorted(test_idx),
            {
                "mode": "stratified_random_fallback",
                "seed": seed,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "n_samples": len(samples),
                "n_groups": len(groups),
            },
        )

    rng = _rng(seed)
    groups_arr = np.array(groups)
    rng.shuffle(groups_arr)

    n_groups = len(groups_arr)
    n_test_g = max(1, int(round(test_ratio * n_groups))) if test_ratio > 0 else 0
    n_val_g = max(1, int(round(val_ratio * n_groups))) if val_ratio > 0 else 0

    # Avoid overlap and keep at least one train group
    n_test_g = min(n_test_g, n_groups - 2)
    n_val_g = min(n_val_g, n_groups - n_test_g - 1)

    test_groups = set(groups_arr[:n_test_g].tolist())
    val_groups = set(groups_arr[n_test_g : n_test_g + n_val_g].tolist())

    train_idx, val_idx, test_idx = [], [], []
    for g, ids in by_group.items():
        if g in test_groups:
            test_idx.extend(ids)
        elif g in val_groups:
            val_idx.extend(ids)
        else:
            train_idx.extend(ids)

    summary = {
        "mode": "group_split",
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_samples": len(samples),
        "n_groups": n_groups,
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
    }

    return sorted(train_idx), sorted(val_idx), sorted(test_idx), summary


def stratified_subsample(
    samples: Sequence[Sample],
    base_indices: Sequence[int],
    fraction: float,
    seed: int,
    min_per_stratum: int = 1,
) -> List[int]:
    """Subsample *within* base_indices while preserving (domain,label) proportions."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    rng = _rng(seed)

    strata: Dict[Tuple[str, float], List[int]] = {}
    for i in base_indices:
        s = samples[i]
        strata.setdefault((s.domain, s.label), []).append(i)

    chosen: List[int] = []
    for _key, ids0 in strata.items():
        ids = np.array(ids0, dtype=int)
        rng.shuffle(ids)
        n = len(ids)
        k = int(round(fraction * n))
        k = max(min_per_stratum, k) if n >= min_per_stratum else n
        k = min(k, n)
        chosen.extend(ids[:k].tolist())

    rng.shuffle(chosen)
    return chosen


# -------------------------
# Normalization (fit on train only)
# -------------------------

def compute_global_mean_std(
    samples: Sequence[Sample],
    indices: Sequence[int],
    seq_len: int,
    feature_cols: Sequence[str],
    clip_policy: str,
    fill_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over (time × samples) on a given index set."""
    n_ch = len(feature_cols)
    total_count = 0
    sum_ = np.zeros((n_ch,), dtype=np.float64)
    sumsq = np.zeros((n_ch,), dtype=np.float64)

    tmp_ds = InsertionCSVJsonDataset(
        samples=samples,
        seq_len=seq_len,
        feature_cols=feature_cols,
        clip_policy=clip_policy,
        fill_value=fill_value,
        normalize="none",
    )

    for idx in indices:
        x, _ = tmp_ds[idx]
        x_np = x.numpy().astype(np.float64)  # [L, C]
        sum_ += x_np.sum(axis=0)
        sumsq += (x_np * x_np).sum(axis=0)
        total_count += x_np.shape[0]

    mean = sum_ / max(total_count, 1)
    var = sumsq / max(total_count, 1) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# -------------------------
# Training loop (best model selection fixed)
# -------------------------

def _format_hms(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"


@torch.no_grad()
def evaluate_silent(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict]:
    """Call tp.evaluate but silence prints."""
    with contextlib.redirect_stdout(None):
        loss, _acc, best_stats = tp.evaluate(model, loader, device)
    return float(loss), dict(best_stats)


def train_single_run(
    *,
    model_preset: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    grad_clip: float,
    seed: int,
    select_metric: str = "bal_acc",  # bal_acc|f1|acc|precision|recall|specificity
    patience: int = 5,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict]:
    """Train from scratch and return best model (by select_metric on val) + summary dict."""
    tp.set_seed(seed)

    model_cfg = get_model_config(model_preset, num_classes=1)
    model = PatchTSTClassifier(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    best_val_stats: Optional[Dict] = None
    best_val_loss: float = float("inf")

    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = tp.train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=amp,
            grad_clip=grad_clip,
        )

        if verbose:
            val_loss, _acc, val_stats = tp.evaluate(model, val_loader, device)
        else:
            val_loss, val_stats = evaluate_silent(model, val_loader, device)

        score = float(val_stats.get(select_metric, float("nan")))
        if math.isnan(score):
            raise RuntimeError(
                f"select_metric '{select_metric}' not found. keys={list(val_stats.keys())}"
            )

        if verbose:
            print(
                f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc@0.5 {tr_acc:.3f} | "
                f"val loss {val_loss:.4f} {select_metric} {score:.3f}"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_loss = float(val_loss)
            best_val_stats = dict(val_stats)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            break

    total_time = time.time() - t0

    if best_state is None or best_val_stats is None:
        raise RuntimeError("Training produced no best state (unexpected).")

    model.load_state_dict(best_state)

    summary = {
        "model_preset": model_preset,
        "model_cfg": config_to_dict(model_cfg),
        "seed": seed,
        "epochs_ran": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_score": best_score,
        "select_metric": select_metric,
        "best_val_stats": best_val_stats,
        "train_time_s": total_time,
    }
    return model, summary


def write_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def append_rows_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    if write_header:
        keys = set()
        for r in rows:
            keys |= set(r.keys())
        fieldnames = sorted(keys)
    else:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            fieldnames = next(reader)

    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_fractions(s: str) -> List[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", type=str, required=True, help="Root folder containing CSV + JSON pairs")
    ap.add_argument("--out-dir", type=str, required=True, help="Output folder for splits + results")

    ap.add_argument("--model-preset", type=str, default="small", choices=["small", "base"], help="PatchTST preset")

    ap.add_argument("--seq-len", type=int, default=1000, help="Fixed length L after trunc/pad")
    ap.add_argument("--features", type=str, default="wrench6", help="Feature preset or comma-separated list of column names")
    ap.add_argument("--clip-policy", type=str, default="end", choices=["start", "end", "center"], help="How to truncate longer sequences")
    ap.add_argument("--normalize", type=str, default="global", choices=["none", "per_sample", "global"], help="Normalization mode")

    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=123, help="Seed for generating benchmark split")
    ap.add_argument("--splits-file", type=str, default="", help="Optional existing splits JSON to reuse")

    ap.add_argument("--group-key", type=str, default="", help="Metadata key to use as session/run group (recommended: session_id)")
    ap.add_argument("--domain-key", type=str, default="", help="Metadata key to use as domain label")

    ap.add_argument("--fractions", type=str, default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--repeats", type=int, default=5, help="#repeats per fraction (different seeds + subsamples)")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--min-per-stratum", type=int, default=5)

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument(
        "--select-metric",
        type=str,
        default="bal_acc",
        choices=["bal_acc", "f1", "acc", "precision", "recall", "specificity"],
        help="Metric used for early stopping / selecting best epoch",
    )

    ap.add_argument("--verbose", action="store_true")

    # Optional OOD protocol
    ap.add_argument("--ood-key", type=str, default="", help="Metadata key for OOD grouping (e.g., insertion_object)")
    ap.add_argument("--ood-values", type=str, default="", help="Comma-separated holdout values. Empty => use all unique values")
    ap.add_argument("--ood-repeats", type=int, default=3)

    args = ap.parse_args(list(argv) if argv is not None else None)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Features
    if args.features.strip().lower() == "wrench6":
        feature_cols = DEFAULT_FEATURES_WRENCH6
    else:
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
        if not feature_cols:
            raise SystemExit("--features is empty")

    group_key = args.group_key.strip() or None
    domain_key = args.domain_key.strip() or None

    samples = index_samples(data_root, group_key=group_key, domain_key=domain_key)
    print(f"Indexed {len(samples)} samples")
    print(f"Unique groups: {len({s.group for s in samples})}")
    print(f"Unique domains: {len({s.domain for s in samples})}")

    # --- Load or create fixed IID split
    splits_path = Path(args.splits_file) if args.splits_file else (out_dir / "splits_iid.json")
    if splits_path.exists():
        split_obj = json.loads(splits_path.read_text())
        train_idx = list(map(int, split_obj["train_idx"]))
        val_idx = list(map(int, split_obj["val_idx"]))
        test_idx = list(map(int, split_obj["test_idx"]))
        split_summary = split_obj.get("summary", {})
        print(f"Loaded existing IID split: {splits_path}")
    else:
        train_idx, val_idx, test_idx, split_summary = group_split(
            samples,
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.split_seed),
        )
        write_json(
            splits_path,
            {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx, "summary": split_summary},
        )
        print(f"Wrote IID split: {splits_path}")

    # --- Fit global normalization on *full IID train* once
    global_mean = global_std = None
    if args.normalize == "global":
        print("Computing global mean/std on IID train ...")
        global_mean, global_std = compute_global_mean_std(
            samples=samples,
            indices=train_idx,
            seq_len=int(args.seq_len),
            feature_cols=feature_cols,
            clip_policy=args.clip_policy,
            fill_value=0.0,
        )
        np.save(out_dir / "global_mean.npy", global_mean)
        np.save(out_dir / "global_std.npy", global_std)
        print("Saved global_mean.npy and global_std.npy")

    base_ds = InsertionCSVJsonDataset(
        samples=samples,
        seq_len=int(args.seq_len),
        feature_cols=feature_cols,
        clip_policy=args.clip_policy,
        fill_value=0.0,
        normalize=args.normalize,
        global_mean=global_mean,
        global_std=global_std,
    )

    val_loader = DataLoader(
        Subset(base_ds, val_idx),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(base_ds, test_idx),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    amp = bool(args.amp) and not bool(args.no_amp)

    # --- Learning curve runs
    fractions = parse_fractions(args.fractions)
    results_csv = out_dir / "learning_curve_results.csv"

    print("\n=== IID Learning curve ===")
    print("fractions:", fractions)
    print("repeats:", args.repeats)

    for f in fractions:
        for rep in range(int(args.repeats)):
            run_seed = int(args.base_seed) + 10_000 * rep + int(round(f * 1000))
            subsample_seed = run_seed + 7

            sub_train_idx = stratified_subsample(
                samples=samples,
                base_indices=train_idx,
                fraction=float(f),
                seed=subsample_seed,
                min_per_stratum=int(args.min_per_stratum),
            )

            train_loader = DataLoader(
                Subset(base_ds, sub_train_idx),
                batch_size=int(args.batch_size),
                shuffle=True,
                num_workers=int(args.num_workers),
                pin_memory=True,
            )

            model, tr_summary = train_single_run(
                model_preset=args.model_preset,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                amp=amp,
                grad_clip=float(args.grad_clip),
                seed=run_seed,
                select_metric=args.select_metric,
                patience=int(args.patience),
                verbose=bool(args.verbose),
            )

            # Evaluate on fixed IID test
            if args.verbose:
                test_loss, _acc, test_stats = tp.evaluate(model, test_loader, device)
            else:
                test_loss, test_stats = evaluate_silent(model, test_loader, device)

            row = {
                "phase": "iid_learning_curve",
                "fraction": float(f),
                "rep": int(rep),
                "run_seed": int(run_seed),
                "subsample_seed": int(subsample_seed),
                "train_samples": int(len(sub_train_idx)),
                "val_samples": int(len(val_idx)),
                "test_samples": int(len(test_idx)),
                "select_metric": args.select_metric,
                "best_epoch": tr_summary["best_epoch"],
                "epochs_ran": tr_summary["epochs_ran"],
                "train_time_s": float(tr_summary["train_time_s"]),
                "train_time_hms": _format_hms(float(tr_summary["train_time_s"])),
                "val_loss": float(tr_summary["best_val_loss"]),
                "val_score": float(tr_summary["best_val_score"]),
                "test_loss": float(test_loss),
            }

            # Flatten stats
            for k, v in test_stats.items():
                row[f"test_{k}"] = v
            for k, v in tr_summary["best_val_stats"].items():
                row[f"val_{k}"] = v

            append_rows_csv(results_csv, [row])

            print(
                f"f={f:.2f} rep={rep} train={len(sub_train_idx)} | "
                f"val {args.select_metric}={row.get('val_'+args.select_metric, float('nan')):.3f} "
                f"test bal_acc={row.get('test_bal_acc', float('nan')):.3f} "
                f"time={row['train_time_hms']}"
            )

    print(f"\nWrote IID learning curve results: {results_csv}")

    # --- Optional: OOD evaluation (true holdout -> train excludes that value)
    if args.ood_key.strip():
        ood_key = args.ood_key.strip()
        print("\n=== OOD evaluation ===")

        if args.ood_values.strip():
            holdouts = [v.strip() for v in args.ood_values.split(",") if v.strip()]
        else:
            holdouts = sorted(
                {str(s.meta.get(ood_key, "")) for s in samples if ood_key in s.meta and s.meta.get(ood_key) is not None}
            )

        if not holdouts:
            print(f"No OOD holdout values found for key '{ood_key}'. Skipping.")
        else:
            ood_csv = out_dir / "ood_results.csv"

            for holdout in holdouts:
                holdout_idx = [i for i, s in enumerate(samples) if str(s.meta.get(ood_key, "")) == holdout]
                holdout_set = set(holdout_idx)
                remain_idx = [i for i in range(len(samples)) if i not in holdout_set]

                # Split remaining into train/val (holdout is the test)
                remain_samples = [samples[i] for i in remain_idx]
                r_train_rel, r_val_rel, _r_test_rel, _r_summary = group_split(
                    remain_samples,
                    val_ratio=float(args.val_ratio),
                    test_ratio=0.0,
                    seed=int(args.split_seed) + 999,
                )
                r_train_idx = [remain_idx[i] for i in r_train_rel]
                r_val_idx = [remain_idx[i] for i in r_val_rel]

                # Fit normalization on OOD-train if using global
                ood_mean = ood_std = None
                if args.normalize == "global":
                    ood_mean, ood_std = compute_global_mean_std(
                        samples=samples,
                        indices=r_train_idx,
                        seq_len=int(args.seq_len),
                        feature_cols=feature_cols,
                        clip_policy=args.clip_policy,
                        fill_value=0.0,
                    )

                ood_ds = InsertionCSVJsonDataset(
                    samples=samples,
                    seq_len=int(args.seq_len),
                    feature_cols=feature_cols,
                    clip_policy=args.clip_policy,
                    fill_value=0.0,
                    normalize=args.normalize,
                    global_mean=ood_mean if ood_mean is not None else global_mean,
                    global_std=ood_std if ood_std is not None else global_std,
                )

                r_val_loader = DataLoader(
                    Subset(ood_ds, r_val_idx),
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    num_workers=int(args.num_workers),
                    pin_memory=True,
                )
                holdout_loader = DataLoader(
                    Subset(ood_ds, holdout_idx),
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    num_workers=int(args.num_workers),
                    pin_memory=True,
                )

                for rep in range(int(args.ood_repeats)):
                    run_seed = int(args.base_seed) + 20_000 + 1000 * rep

                    r_train_loader = DataLoader(
                        Subset(ood_ds, r_train_idx),
                        batch_size=int(args.batch_size),
                        shuffle=True,
                        num_workers=int(args.num_workers),
                        pin_memory=True,
                    )

                    model, _tr_summary = train_single_run(
                        model_preset=args.model_preset,
                        train_loader=r_train_loader,
                        val_loader=r_val_loader,
                        device=device,
                        epochs=int(args.epochs),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        amp=amp,
                        grad_clip=float(args.grad_clip),
                        seed=run_seed,
                        select_metric=args.select_metric,
                        patience=int(args.patience),
                        verbose=bool(args.verbose),
                    )

                    if args.verbose:
                        ood_loss, _acc, ood_stats = tp.evaluate(model, holdout_loader, device)
                    else:
                        ood_loss, ood_stats = evaluate_silent(model, holdout_loader, device)

                    row = {
                        "phase": "ood_holdout",
                        "ood_key": ood_key,
                        "ood_holdout": holdout,
                        "rep": int(rep),
                        "run_seed": int(run_seed),
                        "train_samples": int(len(r_train_idx)),
                        "val_samples": int(len(r_val_idx)),
                        "ood_test_samples": int(len(holdout_idx)),
                        "ood_loss": float(ood_loss),
                    }
                    for k, v in ood_stats.items():
                        row[f"ood_{k}"] = v

                    append_rows_csv(ood_csv, [row])

                    print(
                        f"OOD holdout={holdout} rep={rep} | "
                        f"ood bal_acc={row.get('ood_bal_acc', float('nan')):.3f} "
                        f"f1={row.get('ood_f1', float('nan')):.3f}"
                    )

            print(f"Wrote OOD results: {ood_csv}")

    # Summary
    write_json(
        out_dir / "run_summary.json",
        {
            "data_root": str(data_root),
            "n_samples": len(samples),
            "feature_cols": feature_cols,
            "seq_len": int(args.seq_len),
            "clip_policy": args.clip_policy,
            "normalize": args.normalize,
            "model_preset": args.model_preset,
            "iid_split": split_summary,
            "fractions": fractions,
            "repeats": int(args.repeats),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())