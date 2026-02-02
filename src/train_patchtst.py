# train_patchtst.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
import re
from pathlib import Path
from typing import List, Tuple, Set
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Adjust import if your model file/module name differs
from patchtst import PatchTSTClassifier, PatchTSTConfig


# -------------------------
# Dataset
# -------------------------
USE_COLS = ["FX_Base", "FY_Base", "FZ_Base", "MuX_Base", "MuY_Base", "MuZ_Base"]
LABEL_RE = re.compile(r"_(True|False)\.csv$", re.IGNORECASE)


class ForceCSVFolderDataset(Dataset):
    """
    Loads one time series sample per CSV file.

    Expected:
      - Files under root_dir (recursive)
      - Filename ends with _True.csv or _False.csv
      - CSV has header with required columns

    Returns:
      x: FloatTensor [L, C]  (L=1000, C=6 by default)
      y: FloatTensor []      (0.0 or 1.0) for BCEWithLogitsLoss
    """
    def __init__(self, root_dir: str | Path, seq_len: int = 1000, normalize: bool = True):
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.normalize = normalize

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root_dir not found: {self.root_dir}")

        self.files = self._collect_files(self.root_dir)
        if len(self.files) == 0:
            raise RuntimeError(f"No labeled CSV files found under: {self.root_dir}")

    @staticmethod
    def _collect_files(root_dir: Path) -> List[Path]:
        files: List[Path] = []
        for p in root_dir.rglob("*.csv"):
            if LABEL_RE.search(p.name):
                files.append(p)
        files.sort()
        return files

    @staticmethod
    def label_from_name(filename: str) -> float:
        m = LABEL_RE.search(filename)
        if not m:
            raise ValueError(f"Could not parse label from filename: {filename}")
        return 1.0 if m.group(1).lower() == "true" else 0.0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        y = self.label_from_name(path.name)

        df = pd.read_csv(path, usecols=USE_COLS)

        if df.shape[1] != len(USE_COLS):
            raise ValueError(f"{path}: expected {len(USE_COLS)} cols, got {df.shape[1]}")
        if df.shape[0] != self.seq_len:
            raise ValueError(f"{path}: expected {self.seq_len} rows, got {df.shape[0]}")

        x = df.to_numpy(dtype=np.float32)  # [L, 6]

        # Per-sample per-channel normalization (baseline)
        if self.normalize:
            mu = x.mean(axis=0, keepdims=True)
            sig = x.std(axis=0, keepdims=True) + 1e-6
            x = (x - mu) / sig

        x_t = torch.from_numpy(x)                  # [L, C]
        y_t = torch.tensor(y, dtype=torch.float32) # scalar
        return x_t, y_t


# -------------------------
# Train/Eval
# -------------------------
def _default_ckpt_name() -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"patchtst_force_best_{ts}.pt"


@dataclass
class TrainCfg:
    data_root: str = "ForceDataNovo/Old_Fixture"
    seq_len: int = 1000

    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 25
    amp: bool = True
    grad_clip: float = 1.0
    seed: int = 42

    # Session split
    # Example: {"2025_09_08", "2025_09_09"}
    val_groups: Set[str] = field(default_factory=lambda: {"2025_09_08", "2025_09_09"})

    # Output
    ckpt_dir: str = "checkpoints"
    ckpt_name: str = field(default_factory=_default_ckpt_name)


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


@torch.no_grad()
def confusion_matrix_binary_from_logits(logits: torch.Tensor, y: torch.Tensor, thr: float = 0.5):
    """
    returns TP, FP, TN, FN for positive class y=1
    """
    if logits.dim() == 2 and logits.size(1) == 1:
        logits = logits[:, 0]
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).to(torch.int64)
    y_i = y.to(torch.int64)

    tp = ((pred == 1) & (y_i == 1)).sum().item()
    fp = ((pred == 1) & (y_i == 0)).sum().item()
    tn = ((pred == 0) & (y_i == 0)).sum().item()
    fn = ((pred == 0) & (y_i == 1)).sum().item()
    return int(tp), int(fp), int(tn), int(fn)


def metrics_from_cm(tp: int, fp: int, tn: int, fn: int):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return precision, recall, f1, specificity


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total = 0

    # Collect all logits/labels for threshold sweep
    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)  # [B, L, C]
        y = y.to(device, non_blocking=True)  # [B]

        logits = model(x).squeeze(-1)        # [B]
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)  # [N]
    y = torch.cat(all_y, dim=0)            # [N] float {0,1}

    # Sweep thresholds
    thresholds = torch.linspace(0.05, 0.95, steps=19)  # 0.05, 0.10, ..., 0.95
    best_thr = 0.5
    best_bal_acc = -1.0
    best_stats = None  # (tp, fp, tn, fn, acc, bal_acc, precision, recall, f1, specificity)

    probs = torch.sigmoid(logits)

    for thr in thresholds:
        thr_f = float(thr.item())
        pred = (probs >= thr_f).to(torch.int64)
        y_i = y.to(torch.int64)

        tp = int(((pred == 1) & (y_i == 1)).sum().item())
        fp = int(((pred == 1) & (y_i == 0)).sum().item())
        tn = int(((pred == 0) & (y_i == 0)).sum().item())
        fn = int(((pred == 0) & (y_i == 1)).sum().item())

        # Metrics
        precision, recall, f1, specificity = metrics_from_cm(tp, fp, tn, fn)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        bal_acc = 0.5 * (recall + specificity)  # TPR + TNR / 2

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thr = thr_f
            best_stats = (tp, fp, tn, fn, acc, bal_acc, precision, recall, f1, specificity)

    tp, fp, tn, fn, acc, bal_acc, precision, recall, f1, specificity = best_stats

    print(f"Best threshold by balanced acc: {best_thr:.2f}")
    print(f"Confusion matrix (thr={best_thr:.2f}): TP={tp} FP={fp} TN={tn} FN={fn}")
    print(
        f"Val metrics: Acc={acc:.3f} BalAcc={bal_acc:.3f} "
        f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} Specificity={specificity:.3f}"
    )

    return total_loss / max(total, 1), acc



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

    # New AMP API (removes FutureWarning)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
            logits = model(x).squeeze(-1)  # [B]
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


def main():
    cfg = TrainCfg()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = ForceCSVFolderDataset(cfg.data_root, seq_len=cfg.seq_len, normalize=True)
    print(f"Found {len(ds)} samples under {cfg.data_root}")

    # Group by session folder
    groups = [p.parent.name for p in ds.files]
    print("unique groups:", len(set(groups)))
    print("top groups:", Counter(groups).most_common(10))

    # Build session-based split
    train_idx = [i for i, g in enumerate(groups) if g not in cfg.val_groups]
    val_idx = [i for i, g in enumerate(groups) if g in cfg.val_groups]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    print("train size:", len(train_ds), "val size:", len(val_ds))
    print("val groups:", sorted(cfg.val_groups))

    # Print class balance overall + per split
    all_labels = [ForceCSVFolderDataset.label_from_name(p.name) for p in ds.files]
    train_labels = [ForceCSVFolderDataset.label_from_name(ds.files[i].name) for i in train_idx]
    val_labels = [ForceCSVFolderDataset.label_from_name(ds.files[i].name) for i in val_idx]

    print("overall pos:", sum(all_labels), "neg:", len(all_labels) - sum(all_labels))
    print("train   pos:", sum(train_labels), "neg:", len(train_labels) - sum(train_labels))
    print("val     pos:", sum(val_labels), "neg:", len(val_labels) - sum(val_labels))

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

    # Model
    model_cfg = PatchTSTConfig(
        num_classes=1,            # single logit for BCE
        patch_len=25,
        stride=25,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
        channel_independent=True,
        pooling="mean",
        fuse="mean",
    )
    model = PatchTSTClassifier(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            amp=cfg.amp, grad_clip=cfg.grad_clip
        )
        va_loss, va_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt_path = Path(cfg.ckpt_dir) / cfg.ckpt_name
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg,
                    "epoch": epoch,
                    "val_acc": va_acc,
                    "val_groups": sorted(cfg.val_groups),
                },
                ckpt_path,
            )

    print("Best val acc:", best_val_acc)
    print("Saved best checkpoint as:", str(Path(cfg.ckpt_dir) / cfg.ckpt_name))


if __name__ == "__main__":
    main()
