from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict
from collections import Counter
import os
import json
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from patchtst import PatchTSTClassifier, get_model_config, config_to_dict


# -------------------------
# Dataset helpers
# -------------------------
def _default_ckpt_name() -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"patchtst_force_best_{ts}.pt"


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


# Add/adjust column sets here if your CSV header differs
KNOWN_FEATURE_SETS: List[List[str]] = [
    # UR-style TCP wrench (example from earlier work)
    ["tcp_force_x[N]", "tcp_force_y[N]", "tcp_force_z[N]", "tcp_force_rx[Nm]", "tcp_force_ry[Nm]", "tcp_force_rz[Nm]"],
    # sometimes torque columns are named differently
    ["tcp_force_x[N]", "tcp_force_y[N]", "tcp_force_z[N]", "tcp_torque_x[Nm]", "tcp_torque_y[Nm]", "tcp_torque_z[Nm]"],
    # your previous dataset style (kept as fallback)
    ["FX_Base", "FY_Base", "FZ_Base", "MuX_Base", "MuY_Base", "MuZ_Base"],
]


def infer_feature_cols(csv_path: Path, requested: Optional[List[str]]) -> List[str]:
    header = list(pd.read_csv(csv_path, nrows=0).columns)

    if requested is not None:
        missing = [c for c in requested if c not in header]
        if missing:
            raise ValueError(f"{csv_path}: requested feature columns missing: {missing}\nHeader: {header}")
        return requested

    for cols in KNOWN_FEATURE_SETS:
        if all(c in header for c in cols):
            return cols

    raise ValueError(
        f"{csv_path}: could not auto-detect 6 feature columns.\n"
        f"Header: {header}\n"
        f"Fix: set TrainCfg.feature_cols explicitly to the 6 columns you want."
    )


def clip_or_pad(x: np.ndarray, seq_len: int, clip_policy: str) -> np.ndarray:
    """
    x: [L, C]
    returns [seq_len, C]
    """
    L, C = x.shape
    if L == seq_len:
        return x
    if L > seq_len:
        if clip_policy == "end":
            return x[-seq_len:, :]
        elif clip_policy == "start":
            return x[:seq_len, :]
        else:
            raise ValueError(f"Unknown clip_policy: {clip_policy} (use 'end' or 'start')")
    # pad
    out = np.zeros((seq_len, C), dtype=x.dtype)
    out[:L, :] = x
    return out


class ForceCSVFolderDataset(Dataset):
    """
    One sample per CSV file.

    Expected structure (your case):
      training_data/batch_1/<group_folder>/*.csv
      training_data/batch_1/<group_folder>/*.json  (same base name)
    JSON must contain: successful_insertion: true/false
    """
    def __init__(
        self,
        root_dir: str | Path,
        seq_len: int,
        clip_policy: str = "end",
        normalize: bool = True,
        feature_cols: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.seq_len = int(seq_len)
        self.clip_policy = str(clip_policy)
        self.normalize = bool(normalize)
        self.feature_cols_requested = feature_cols  # can be None (auto)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root_dir not found: {self.root_dir}")

        self.samples: List[Dict] = self._index_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"No usable (csv+json) samples found under: {self.root_dir}")

        # Determine feature columns once (from first file) if not explicit.
        self.feature_cols = infer_feature_cols(self.samples[0]["csv"], self.feature_cols_requested)

    def _index_samples(self) -> List[Dict]:
        out = []
        for csv_path in sorted(self.root_dir.rglob("*.csv")):
            json_path = csv_path.with_suffix(".json")
            if not json_path.exists():
                continue

            meta = _read_json(json_path)
            if "successful_insertion" not in meta:
                raise KeyError(f"{json_path}: missing key 'successful_insertion'")

            y = 1.0 if bool(meta["successful_insertion"]) else 0.0
            group = csv_path.parent.name  # folder = group/session/domain

            out.append({"csv": csv_path, "json": json_path, "y": y, "group": group})

        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        csv_path: Path = s["csv"]
        y: float = float(s["y"])

        # auto-detect (or validate) feature cols for THIS file too (in case headers differ)
        cols = infer_feature_cols(csv_path, self.feature_cols_requested)

        df = pd.read_csv(csv_path, usecols=cols)
        x = df.to_numpy(dtype=np.float32)  # [L, C]

        x = clip_or_pad(x, seq_len=self.seq_len, clip_policy=self.clip_policy)

        # Per-sample per-channel normalization
        if self.normalize:
            mu = x.mean(axis=0, keepdims=True)
            sig = x.std(axis=0, keepdims=True) + 1e-6
            x = (x - mu) / sig

        x_t = torch.from_numpy(x)                  # [seq_len, C]
        y_t = torch.tensor(y, dtype=torch.float32) # scalar
        return x_t, y_t


# -------------------------
# Train/Eval
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> Tuple[int, int]:
    if logits.dim() == 2 and logits.size(1) == 1:
        logits = logits[:, 0]
    pred = (torch.sigmoid(logits) >= thr).to(y.dtype)
    correct = (pred == y).sum().item()
    total = y.numel()
    return int(correct), int(total)


def metrics_from_cm(tp: int, fp: int, tn: int, fn: int):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return precision, recall, f1, specificity


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, dict]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total = 0
    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).squeeze(-1)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    thresholds = torch.linspace(0.05, 0.95, steps=19)
    best_thr = 0.5
    best_bal_acc = -1.0
    best_stats = None

    probs = torch.sigmoid(logits)

    for thr in thresholds:
        thr_f = float(thr.item())
        pred = (probs >= thr_f).to(torch.int64)
        y_i = y.to(torch.int64)

        tp = int(((pred == 1) & (y_i == 1)).sum().item())
        fp = int(((pred == 1) & (y_i == 0)).sum().item())
        tn = int(((pred == 0) & (y_i == 0)).sum().item())
        fn = int(((pred == 0) & (y_i == 1)).sum().item())

        precision, recall, f1, specificity = metrics_from_cm(tp, fp, tn, fn)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        bal_acc = 0.5 * (recall + specificity)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thr = thr_f
            best_stats = (tp, fp, tn, fn, acc, bal_acc, precision, recall, f1, specificity)

    tp, fp, tn, fn, acc, bal_acc, precision, recall, f1, specificity = best_stats

    stats = {
        "selection_metric": "balanced_accuracy",
        "best_threshold": float(best_thr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
    }

    print(f"Best threshold by balanced acc: {best_thr:.2f}")
    print(f"Confusion matrix (thr={best_thr:.2f}): TP={tp} FP={fp} TN={tn} FN={fn}")
    print(
        f"Val metrics: Acc={acc:.3f} BalAcc={bal_acc:.3f} "
        f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} Specificity={specificity:.3f}"
    )

    return total_loss / max(total, 1), float(acc), stats


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu",
        enabled=(amp and device.type == "cuda")
    )

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        c, t = binary_accuracy_from_logits(logits.detach(), y, thr=0.5)
        correct += c
        total += t

    return total_loss / max(total, 1), correct / max(total, 1)


@dataclass
class TrainCfg:
    model_preset: str = "big"

    # Your new structure
    data_root: str = "training_data/batch_1"
    seq_len: int = 2000
    clip_policy: str = "end"  # "end" keeps the last seq_len rows (good if the event is near the end)

    # If None: auto-detect from header. Otherwise set explicitly to your 6 columns.
    # feature_cols: Optional[List[str]] = None
    feature_cols = ["tcp_force_x[N]", "tcp_force_y[N]","tcp_force_z[N]","tcp_force_rx[Nm?]","tcp_force_ry[Nm?]","tcp_force_rz[Nm?]"]

    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 25
    amp: bool = True
    grad_clip: float = 1.0
    seed: int = 42

    # Validation split by folder name:
    # - if empty, auto-pick 1 group deterministically from available groups
    # val_groups: Set[str] = field(default_factory=set)
    val_groups: Set[str] = field(default_factory=lambda: {"rect_rod_t0.3"})

    # Output
    ckpt_dir: str = "checkpoints"
    ckpt_name: str = field(default_factory=_default_ckpt_name)


def main():
    cfg = TrainCfg()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = ForceCSVFolderDataset(
        cfg.data_root,
        seq_len=cfg.seq_len,
        clip_policy=cfg.clip_policy,
        normalize=True,
        feature_cols=cfg.feature_cols,
    )
    print(f"Found {len(ds)} samples under {cfg.data_root}")
    print("Using feature columns:", ds.feature_cols)

    groups = [s["group"] for s in ds.samples]
    uniq_groups = sorted(set(groups))
    print("unique groups:", len(uniq_groups))
    print("top groups:", Counter(groups).most_common(10))

    # Auto-pick one val group if none specified
    if len(cfg.val_groups) == 0:
        rng = np.random.default_rng(cfg.seed)
        chosen = rng.choice(uniq_groups, size=1, replace=False).tolist()
        cfg.val_groups = set(chosen)
        print("Auto-selected val_groups:", sorted(cfg.val_groups))

    train_idx = [i for i, g in enumerate(groups) if g not in cfg.val_groups]
    val_idx = [i for i, g in enumerate(groups) if g in cfg.val_groups]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    print("train size:", len(train_ds), "val size:", len(val_ds))
    print("val groups:", sorted(cfg.val_groups))

    # Class balance
    all_labels = [float(s["y"]) for s in ds.samples]
    train_labels = [float(ds.samples[i]["y"]) for i in train_idx]
    val_labels = [float(ds.samples[i]["y"]) for i in val_idx]
    print("overall pos:", int(sum(all_labels)), "neg:", int(len(all_labels) - sum(all_labels)))
    print("train   pos:", int(sum(train_labels)), "neg:", int(len(train_labels) - sum(train_labels)))
    print("val     pos:", int(sum(val_labels)), "neg:", int(len(val_labels) - sum(val_labels)))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model_cfg = get_model_config(cfg.model_preset, num_classes=1)
    print("MODEL CFG:", model_cfg)

    model = PatchTSTClassifier(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    best_val_acc = -1.0
    best_epoch = None
    best_val_loss = None
    best_stats = None
    best_state_dict = None
    best_time_to_best_s = None

    train_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        print()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            amp=cfg.amp, grad_clip=cfg.grad_clip
        )
        va_loss, va_acc, va_best_stats = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
            f"time {epoch_time:.2f}s"
        )

        # Save BEST (fixes bug from your previous version)
        if va_acc > best_val_acc:
            best_val_acc = float(va_acc)
            best_val_loss = float(va_loss)
            best_epoch = int(epoch)
            best_stats = va_best_stats
            best_state_dict = copy.deepcopy(model.state_dict())
            best_time_to_best_s = float(time.time() - train_start)

    total_train_time_s = float(time.time() - train_start)

    ckpt = {
        "model_state": best_state_dict,
        "model_cfg": config_to_dict(model_cfg),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_stats": best_stats,
        "val_groups": sorted(cfg.val_groups),

        "feature_cols": ds.feature_cols,
        "seq_len": int(cfg.seq_len),
        "clip_policy": cfg.clip_policy,

        "time_to_best_s": best_time_to_best_s,
        "total_train_time_s": total_train_time_s,
        "epochs_ran": int(cfg.epochs),
    }

    ckpt_path = Path(cfg.ckpt_dir) / cfg.ckpt_name
    torch.save(ckpt, ckpt_path)

    print("Saved checkpoint:", ckpt_path)
    print("time_to_best_s:", best_time_to_best_s, "total_train_time_s:", total_train_time_s)


if __name__ == "__main__":
    main()