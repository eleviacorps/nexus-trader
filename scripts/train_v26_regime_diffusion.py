"""V26 Phase 1 Agent 3: Regime-Aware Diffusion Fine-Tuning

Fine-tunes the Phase 0.7 diffusion model with regime conditioning.
Loads from: models/v24/diffusion_unet1d_v2_6m_phase07.pt
Saves to:   models/v26/diffusion_unet1d_v2_regime.pt

Key changes:
- Uses RegimeDiffusionDataset with pre-computed regime labels
- Adds regime embedding layer to combine temporal + regime conditioning
- Fine-tunes at lower learning rate (5e-6) for stability
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    V24_DIFFUSION_FUSED_6M_PATH,
    V24_DIFFUSION_TIMESTAMPS_6M_PATH,
    OUTPUTS_V24_DIR,
)
from src.v24.diffusion.dataset import split_by_year
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.temporal_encoder import TemporalEncoder
from src.v24.diffusion.unet_1d import DiffusionUNet1D
from src.v26.diffusion.regime_dataset import RegimeDiffusionDataset
from src.v6.regime_detection import REGIME_LABELS


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)
        self.backup = {}


class RegimeEmbedding(nn.Module):
    """Regime embedding layer for V26 regime conditioning.

    Projects regime probability distribution to an embedding space
    that can be combined with temporal embeddings.

    Args:
        num_regimes: Number of regime labels.
        embed_dim: Output embedding dimension.
    """

    def __init__(self, num_regimes: int = 10, embed_dim: int = 16) -> None:
        super().__init__()
        self.num_regimes = num_regimes
        self.embed_dim = embed_dim

        # Regime embedding: projects (B, num_regimes) -> (B, embed_dim)
        self.regime_net = nn.Sequential(
            nn.Linear(num_regimes, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """Convert regime probabilities to embeddings.

        Args:
            regime_probs: (B, num_regimes) tensor of regime probabilities.

        Returns:
            (B, embed_dim) regime embeddings.
        """
        return self.regime_net(regime_probs)


def _train_one_epoch(
    model,
    temporal_encoder,
    regime_embedding,
    scheduler,
    dataloader,
    dataset,
    optimizer,
    device,
    use_amp,
    acf_w=0.10,
    vol_w=0.10,
    std_w=0.05,
):
    """Train one epoch with regime conditioning."""
    model.train()
    temporal_encoder.train()
    regime_embedding.train()
    dataset.new_epoch()

    total_loss = total_diff = total_acf = total_vol = total_std = 0.0
    n_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (window, past_ctx, regime_probs) in enumerate(dataloader):
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)
        regime_probs = regime_probs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            # Encode temporal context
            temporal_seq, temporal_emb, _ = temporal_encoder(past_ctx)

            # Encode regime
            regime_emb = regime_embedding(regime_probs)

            # Context is the last timestep of past_ctx
            context = past_ctx[:, -1, :]

            # Training loss with regime conditioning
            loss_dict = scheduler.training_loss_regime(
                model, window, context,
                temporal_seq=temporal_seq,
                temporal_emb=temporal_emb,
                regime_emb=regime_emb,
                acf_weight=acf_w,
                vol_weight=vol_w,
                std_weight=std_w,
            )
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Clip gradients for all parameters
        all_params = (
            list(model.parameters()) +
            list(temporal_encoder.parameters()) +
            list(regime_embedding.parameters())
        )
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_diff += loss_dict["diffusion"].item()
        total_acf += loss_dict["acf"].item()
        total_vol += loss_dict["vol"].item()
        total_std += loss_dict["std"].item()
        n_batches += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}, "
                  f"diff={loss_dict['diffusion'].item():.4f}, "
                  f"acf={loss_dict['acf'].item():.4f}, "
                  f"vol={loss_dict['vol'].item():.4f}, "
                  f"std={loss_dict['std'].item():.4f}")

    return {
        "total": total_loss / max(n_batches, 1),
        "diffusion": total_diff / max(n_batches, 1),
        "acf": total_acf / max(n_batches, 1),
        "vol": total_vol / max(n_batches, 1),
        "std": total_std / max(n_batches, 1),
    }


@torch.no_grad()
def _validate(model, temporal_encoder, regime_embedding, scheduler, dataloader, device, use_amp):
    """Validate model on validation set."""
    model.eval()
    temporal_encoder.eval()
    regime_embedding.eval()

    total = 0.0
    n = 0
    for window, past_ctx, regime_probs in dataloader:
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)
        regime_probs = regime_probs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            temporal_seq, temporal_emb, _ = temporal_encoder(past_ctx)
            regime_emb = regime_embedding(regime_probs)
            context = past_ctx[:, -1, :]
            loss = scheduler.training_loss(
                model, window, context,
                temporal_seq=temporal_seq,
                temporal_emb=torch.cat([temporal_emb, regime_emb], dim=-1),
            )
        total += loss.item()
        n += 1

    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="V26 Phase 1 — Regime Diffusion Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--acf-weight", type=float, default=0.10)
    parser.add_argument("--vol-weight", type=float, default=0.10)
    parser.add_argument("--std-weight", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--regime-embed-dim", type=int, default=16)
    parser.add_argument("--ckpt-in", type=str, default="models/v24/diffusion_unet1d_v2_6m_phase07.pt")
    parser.add_argument("--ckpt-out", type=str, default="models/v26/diffusion_unet1d_v2_regime.pt")
    parser.add_argument("--regime-cache", type=str, default="outputs/v26/regime_cache.npy")
    args = parser.parse_args()

    _set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Device: {device}")
    print(f"V26 Phase 1 Regime Fine-Tuning")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  ACF weight: {args.acf_weight}")
    print(f"  Vol weight: {args.vol_weight}")
    print(f"  Std weight: {args.std_weight}")
    print(f"  Regime embed dim: {args.regime_embed_dim}")
    print(f"  Input checkpoint: {args.ckpt_in}")
    print(f"  Output checkpoint: {args.ckpt_out}")

    # Load Phase 0.7 checkpoint
    ckpt_path = Path(args.ckpt_in)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('best_val_loss', '?'):.6f}")

    # Setup data paths
    fused_path = V24_DIFFUSION_FUSED_6M_PATH
    timestamps_path = V24_DIFFUSION_TIMESTAMPS_6M_PATH
    regime_cache_path = Path(args.regime_cache)
    out_ckpt_path = Path(args.ckpt_out)
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    regime_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load timestamps for year-based split
    timestamps = None
    if timestamps_path.exists():
        timestamps = np.load(str(timestamps_path), mmap_mode="r")

    # Get dataset stats for splitting
    fused_mmap = np.load(str(fused_path), mmap_mode="r")
    total = len(fused_mmap)
    print(f"\nDataset: {total} rows")

    # Split by year
    train_slice, val_slice, _ = split_by_year(total, 120, timestamps=timestamps)
    print(f"  Train: {len(train_slice)}, Val: {len(val_slice)}")

    # Create regime-aware datasets
    print("\nCreating datasets...")
    train_ds = RegimeDiffusionDataset(
        fused_path, 120, train_slice,
        timestamp_path=timestamps_path,
        context_len=256,
        max_samples=args.max_samples,
        load_to_ram=True,
        regime_cache_path=regime_cache_path if not regime_cache_path.exists() else regime_cache_path,
    )
    val_ds = RegimeDiffusionDataset(
        fused_path, 120, val_slice,
        timestamp_path=timestamps_path,
        context_len=256,
        max_samples=min(50000, len(val_slice)),
        load_to_ram=True,
        regime_cache_path=regime_cache_path if regime_cache_path.exists() else None,
    )

    # Print regime distribution
    print("\nRegime distribution (training set):")
    regime_dist = train_ds.get_regime_distribution()
    for regime, pct in sorted(regime_dist.items(), key=lambda x: -x[1]):
        print(f"  {regime}: {pct:.2%}")

    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize models
    num_regimes = len(REGIME_LABELS)
    temporal_dim = 256 + args.regime_embed_dim  # temporal_emb (256) + regime_emb (16)

    print(f"\nInitializing models...")
    print(f"  Number of regimes: {num_regimes}")
    print(f"  Combined temporal dim: {temporal_dim}")

    # U-Net with expanded temporal_dim for regime conditioning
    model = DiffusionUNet1D(
        in_channels=144,
        base_channels=128,
        channel_multipliers=(1, 2, 4),
        time_dim=256,
        num_res_blocks=2,
        ctx_dim=144,
        temporal_dim=temporal_dim,  # Expanded for regime
        d_gru=256,
    ).to(device)

    # Load base model weights from checkpoint
    # Note: EMA weights may need adjustment for expanded temporal_dim
    base_state = ckpt.get("ema", ckpt.get("model", {}))

    # Try to load with strict=False to handle temporal_dim mismatch
    try:
        model.load_state_dict(base_state, strict=False)
        print("  Loaded base model weights (with temporal_dim expansion)")
    except RuntimeError as e:
        print(f"  Warning: Could not load all weights: {e}")
        print("  Initializing from scratch for mismatched layers")

    # Temporal encoder (same as Phase 0.7)
    temporal_encoder = TemporalEncoder(
        in_features=144,
        d_gru=256,
        num_layers=2,
        film_dim=256,
    ).to(device)

    if "temporal_encoder" in ckpt:
        temporal_encoder.load_state_dict(ckpt["temporal_encoder"])
        print("  Loaded temporal encoder weights")

    # New regime embedding layer
    regime_embedding = RegimeEmbedding(
        num_regimes=num_regimes,
        embed_dim=args.regime_embed_dim,
    ).to(device)

    print("  Initialized regime embedding layer")

    # Scheduler and optimizer
    scheduler = NoiseScheduler(1000).to(device)

    optimizer = AdamW(
        list(model.parameters()) +
        list(temporal_encoder.parameters()) +
        list(regime_embedding.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    ema = EMA(model, decay=0.999)

    # Setup logging
    log_path = OUTPUTS_V24_DIR / "v26_regime_diffusion_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training...")
    print(f"  Log file: {log_path}")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train one epoch
        loss_dict = _train_one_epoch(
            model, temporal_encoder, regime_embedding, scheduler,
            train_dl, train_ds, optimizer, device, use_amp,
            args.acf_weight, args.vol_weight, args.std_weight,
        )

        # Update EMA
        ema.update(model)

        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_total": loss_dict["total"],
            "train_diff": loss_dict["diffusion"],
            "train_acf": loss_dict["acf"],
            "train_vol": loss_dict["vol"],
            "train_std": loss_dict["std"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }

        # Validate every epoch
        ema.apply(model)
        val_loss = _validate(model, temporal_encoder, regime_embedding, scheduler, val_dl, device, use_amp)
        ema.restore(model)

        entry["val_loss"] = val_loss

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            entry["best"] = True

            # Save checkpoint with all components
            torch.save({
                "model": model.state_dict(),
                "temporal_encoder": temporal_encoder.state_dict(),
                "regime_embedding": regime_embedding.state_dict(),
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "phase": "v26_1_regime",
                "config": {
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "regime_embed_dim": args.regime_embed_dim,
                },
            }, str(out_ckpt_path))
            print(f"  ** SAVED BEST ** val_loss={val_loss:.6f}")

        # Log and step scheduler
        lr_scheduler.step()
        with open(str(log_path), "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"Epoch {epoch}: total={loss_dict['total']:.4f}, "
              f"diff={loss_dict['diffusion']:.4f}, "
              f"acf={loss_dict['acf']:.4f}, "
              f"vol={loss_dict['vol']:.4f}, "
              f"std={loss_dict['std']:.4f}, "
              f"val={val_loss:.4f} | {elapsed:.0f}s")

    print(f"\nTraining complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoint saved to: {out_ckpt_path}")


if __name__ == "__main__":
    main()
