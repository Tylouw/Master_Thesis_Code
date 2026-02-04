# patchtst_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchTSTConfig:
    # Data-related (fill in later if you want strict checks)
    seq_len: Optional[int] = None          # L (optional; model can infer at runtime)
    n_channels: Optional[int] = None       # C (optional; can infer at runtime if provided at init)
    num_classes: int = 2

    # Patching
    patch_len: int = 16                    # P
    stride: int = 16                       # S (set < patch_len for overlap)
    pad_end: bool = True                   # pad sequence end to fit an integer number of patches

    # Model
    d_model: int = 128 #embedding size bzw token dimension (128)
    n_heads: int = 8 #(8)
    n_layers: int = 4
    d_ff: int = 256 # (256)
    dropout: float = 0.1
    activation: Literal["gelu", "relu"] = "gelu"

    # Strategy
    channel_independent: bool = True       # CI encoder (per channel) vs mixing channels in tokens
    pooling: Literal["cls", "mean"] = "mean"
    fuse: Literal["mean", "mlp"] = "mean"  # how to fuse channels when channel_independent=True


class Patchify(nn.Module):
    """
    Converts x: [B, L, C] into patches.
    Returns:
      patches: [B, N, P, C]
      N: number of patches
    """
    def __init__(self, patch_len: int, stride: int, pad_end: bool = True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad_end = pad_end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x as [B, L, C], got shape {tuple(x.shape)}")

        B, L, C = x.shape
        P, S = self.patch_len, self.stride

        if self.pad_end:
            # Pad so (L - P) is divisible by S, allowing an integer number of patches.
            # Compute target length L_pad such that N = floor((L_pad - P)/S)+1 is integer naturally.
            if L < P:
                L_pad = P
            else:
                remainder = (L - P) % S
                L_pad = L if remainder == 0 else L + (S - remainder)

            if L_pad != L:
                pad_amt = L_pad - L
                # pad along length dimension (dim=1): pad format is (last_dim_left, last_dim_right, ..., dim1_left, dim1_right)
                x = F.pad(x, (0, 0, 0, pad_amt), mode="constant", value=0.0)
                L = L_pad
        else:
            if L < P:
                raise ValueError(f"Sequence length L={L} smaller than patch_len P={P} with pad_end=False")

        # Unfold along length dimension: [B, L, C] -> [B, N, P, C]
        # torch.Tensor.unfold for dim=1
        patches = x.unfold(dimension=1, size=P, step=S)  # [B, N, P, C]
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned position embedding for patch tokens.
    Created lazily if seq_len is unknown; we infer N at first forward pass.
    """
    def __init__(self, d_model: int, max_patches: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_patches = max_patches
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(max_patches, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, N, D]
        N = x.size(-2)
        if N > self.max_patches:
            raise ValueError(f"N={N} exceeds max_patches={self.max_patches}. Increase max_patches.")
        idx = torch.arange(N, device=x.device)
        pe = self.pos_emb(idx)  # [N, D]
        return self.dropout(x + pe)


def _get_activation(name: str):
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, activation: str):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,   # input as [B, N, D]
            norm_first=True,    # Pre-LN tends to be stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, D]
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

def patchtst_base(num_classes: int = 1) -> PatchTSTConfig:
    return PatchTSTConfig(
        num_classes=num_classes,
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
        # keep any other fields you already have
    )

def patchtst_small(num_classes: int = 1) -> PatchTSTConfig:
    return PatchTSTConfig(
        num_classes=num_classes,
        patch_len=25,
        stride=25,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        channel_independent=True,
        pooling="mean",
        fuse="mean",
    )

MODEL_PRESETS = {
    "base": patchtst_base,
    "small": patchtst_small,
}

def get_model_config(preset: str, num_classes: int = 1) -> PatchTSTConfig:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[preset](num_classes=num_classes)

def config_to_dict(cfg: PatchTSTConfig) -> dict:
    return asdict(cfg)

def config_from_dict(d: dict) -> PatchTSTConfig:
    return PatchTSTConfig(**d)


class PatchTSTClassifier(nn.Module):
    """
    PatchTST template classifier.

    Expected input:
      x: [B, L, C]
    Output:
      logits: [B, num_classes]
    """
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg

        self.patchify = Patchify(cfg.patch_len, cfg.stride, cfg.pad_end)

        self.use_cls = (cfg.pooling == "cls")
        self.channel_independent = cfg.channel_independent

        # Tokenization:
        # - channel_independent: each channel patch [P] -> D (shared projection) and encode over N
        # - channel_mixed: each patch [P*C] -> D
        if cfg.channel_independent:
            self.patch_proj = nn.Linear(cfg.patch_len, cfg.d_model)
        else:
            # In channel-mixed mode we need n_channels known at init OR infer later.
            # We'll implement a lazy linear so you can instantiate without knowing C.
            self.patch_proj = nn.LazyLinear(cfg.d_model)

        self.pos_emb = LearnedPositionalEmbedding(cfg.d_model, max_patches=4096, dropout=cfg.dropout)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        else:
            self.cls_token = None

        self.encoder = TransformerEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )

        # Fusion across channels (only used if channel_independent=True)
        if cfg.channel_independent and cfg.fuse == "mlp":
            self.channel_fuse = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model),
                _get_activation(cfg.activation),
                nn.Dropout(cfg.dropout),
            )
        else:
            self.channel_fuse = None

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            _get_activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

        self._init_parameters()

    def _init_parameters(self):
        # Initialize CLS token if used
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x as [B, L, C], got shape {tuple(x.shape)}")
        B, L, C = x.shape

        patches = self.patchify(x)          # [B, N, P, C]
        B, N, P, C = patches.shape

        if self.channel_independent:
            # [B, N, P, C] -> [B, C, N, P]
            patches = patches.permute(0, 3, 1, 2).contiguous()

            # Project P->D for each channel, shared weights:
            # reshape to merge B and C for encoder efficiency
            patches = patches.view(B * C, N, P)       # [B*C, N, P]
            tokens = self.patch_proj(patches)         # [B*C, N, D]

            # Add CLS token if requested
            if self.use_cls:
                cls = self.cls_token.expand(B * C, -1, -1)  # [B*C, 1, D]
                tokens = torch.cat([cls, tokens], dim=1)    # [B*C, 1+N, D]

            tokens = self.pos_emb(tokens)             # + position
            z = self.encoder(tokens)                  # [B*C, 1+N, D] or [B*C, N, D]

            # Pool over patch tokens
            if self.use_cls:
                rep = z[:, 0, :]                      # [B*C, D]
            else:
                rep = z.mean(dim=1)                   # [B*C, D]

            rep = rep.view(B, C, -1)                  # [B, C, D]

            # Fuse across channels
            if self.cfg.fuse == "mean":
                rep = rep.mean(dim=1)                 # [B, D]
            elif self.cfg.fuse == "mlp":
                rep = rep.mean(dim=1)                 # [B, D] (simple start; can replace with attention later)
                rep = self.channel_fuse(rep)          # [B, D]
            else:
                raise ValueError(f"Unknown fuse mode: {self.cfg.fuse}")

        else:
            # Channel-mixed tokenization:
            # [B, N, P, C] -> [B, N, P*C]
            tokens = patches.reshape(B, N, P * C)      # [B, N, P*C]
            tokens = self.patch_proj(tokens)           # [B, N, D] (LazyLinear infers input dim)

            if self.use_cls:
                cls = self.cls_token.expand(B, -1, -1)     # [B, 1, D]
                tokens = torch.cat([cls, tokens], dim=1)   # [B, 1+N, D]

            tokens = self.pos_emb(tokens)
            z = self.encoder(tokens)

            if self.use_cls:
                rep = z[:, 0, :]                      # [B, D]
            else:
                rep = z.mean(dim=1)                   # [B, D]

        logits = self.head(rep)                        # [B, num_classes]
        return logits


if __name__ == "__main__":
    # Smoke test with placeholder sizes
    cfg = PatchTSTConfig(
        seq_len=None,
        n_channels=None,
        num_classes=5,
        patch_len=16,
        stride=16,
        d_model=128,
        n_heads=8,
        n_layers=4,
        channel_independent=True,
        pooling="mean",
        fuse="mean",
    )

    model = PatchTSTClassifier(cfg)

    # Example dummy input: [B, L, C]
    x = torch.randn(4, 256, 6)
    y = model(x)
    print("logits shape:", y.shape)  # [4, 5]
