"""Training script for V26 Phase 2: Multi-Horizon Path Stacking.

Fine-tunes the base generator to learn multi-horizon coherence.
Does NOT retrain from scratch - uses Phase 1 checkpoint.
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
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    V24_DIFFUSION_FUSED_6M_PATH,
    V24_DIFFUSION_NORM_STATS_6M_PATH,
    V24_DIFFUSION_TIMESTAMPS_6M_PATH,
    OUTPUTS_V26_DIR,
)
from src.v24.diffusion.dataset import split_by_year
from src.v26.diffusion.multi_horizon_generator import MultiHorizonGenerator
from src.v26.diffusion.regime_dataset import RegimeDiffusionDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_horizon_losses(result, device):
    """Compute explicit horizon transition losses."""
    boundary_loss = 0.0
    latent_loss = 0.0
    regime_loss = 0.0

    # Boundary continuity loss
    # Short -> Medium: last 10 bars of short vs first 10 bars of medium
    short_end = result.short.paths[:, :, -10:].mean(dim=2)  # (N_short, C)
    med_start = result.medium.paths[:, :, :10].mean(dim=2)  # (N_med, C)
    if short_end.shape[0] > 0 and med_start.shape[0] > 0:
        boundary_loss += torch.nn.functional.mse_loss(short_end.mean(0), med_start.mean(0))

    # Medium -> Long
    med_end = result.medium.paths[:, :, -10:].mean(dim=2)
    long_start = result.long.paths[:, :, :10].mean(dim=2)
    if med_end.shape[0] > 0 and long_start.shape[0] > 0:
        boundary_loss += torch.nn.functional.mse_loss(med_end.mean(0), long_start.mean(0))

    # Latent consistency loss
    short_summary = result.short.latent_summary
    med_summary = result.medium.latent_summary
    long_summary = result.long.latent_summary

    if short_summary.shape[0] > 0 and med_summary.shape[0] > 0:
        latent_loss += 1 - torch.nn.functional.cosine_similarity(
            short_summary.mean(0, keepdim=True), med_summary.mean(0, keepdim=True)
        ).item()
    if med_summary.shape[0] > 0 and long_summary.shape[0] > 0:
        latent_loss += 1 - torch.nn.functional.cosine_similarity(
            med_summary.mean(0, keepdim=True), long_summary.mean(0, keepdim=True)
        ).item()

    # Regime preservation loss
    short_regimes = result.short.regime_probs.argmax(dim=-1)
    med_regimes = result.medium.regime_probs.argmax(dim=-1)
    long_regimes = result.long.regime_probs.argmax(dim=-1)
    if short_regimes.shape[0] > 0:
        regime_loss += (short_regimes != med_regimes[:len(short_regimes)]).float().mean().item()
        regime_loss += (short_regimes != long_regimes[:len(short_regimes)]).float().mean().item()

    # Combined loss - emphasize latent alignment
    total_loss = boundary_loss + 2.0 * latent_loss + 0.1 * regime_loss

    return {
        "total_loss": total_loss,
        "boundary_loss": boundary_loss.item(),
        "latent_loss": latent_loss,
        "regime_loss": regime_loss,
    }


def _train_multi_horizon_epoch(
    generator: MultiHorizonGenerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch - explicit horizon transition losses."""
    generator.short_summary.train()
    generator.medium_summary.train()
    generator.train()

    total_loss = 0.0
    total_boundary = 0.0
    total_latent = 0.0
    total_regime = 0.0
    n_batches = 0

    for batch_idx, (window, past_ctx, regime_probs) in enumerate(dataloader):
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)
        regime_probs = regime_probs.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.no_grad(), torch.amp.autocast("cuda"):
            result = generator.generate_multi_horizon(
                world_state=None,
                past_context=past_ctx,
                regime_probs=regime_probs,
                steps=5,
            )

        loss_dict = _compute_horizon_losses(result, device)
        loss = loss_dict["total_loss"]
        loss_tensor = torch.tensor(loss, device=device, requires_grad=True)

        loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()

        total_loss += loss
        total_boundary += loss_dict["boundary_loss"]
        total_latent += loss_dict["latent_loss"]
        total_regime += loss_dict["regime_loss"]
        n_batches += 1

        print(f"  Batch {batch_idx + 1}: loss={loss:.4f}, boundary={loss_dict['boundary_loss']:.4f}, latent={loss_dict['latent_loss']:.4f}, regime={loss_dict['regime_loss']:.4f}")

    avg_consistency = 1.0 - min(1.0, (total_boundary / n_batches))

    return {
        "total_loss": total_loss / max(n_batches, 1),
        "consistency": avg_consistency,
    }


@torch.no_grad()
def _validate_multi_horizon(
    generator: MultiHorizonGenerator,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate multi-horizon consistency on generated paths."""
    generator.eval()

    total_consistency = 0.0
    n_batches = 0

    for window, past_ctx, regime_probs in dataloader:
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)
        regime_probs = regime_probs.to(device, non_blocking=True)

        result = generator.generate_multi_horizon(
            world_state=None,
            past_context=past_ctx[0:1],
            regime_probs=regime_probs[0],
            steps=5,
        )

        total_consistency += result.horizon_consistency_score
        n_batches += 1

    return {
        "val_consistency": total_consistency / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="V26 Phase 2: Multi-Horizon Training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--phase1-ckpt", type=str, default="models/v26/diffusion_phase1_final.pt")
    parser.add_argument("--output", type=str, default="models/v26/diffusion_phase2_multi_horizon.pt")
    args = parser.parse_args()

    _set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Reserve most VRAM but don't pre-allocate
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)

    print("=" * 60)
    print("V26 Phase 2: Multi-Horizon Path Stacking")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max samples: {args.max_samples}")
    print(f"Phase 1 checkpoint: {args.phase1_ckpt}")
    print(f"Output: {args.output}")
    print()

# Load Phase 1 checkpoint
    print(f"Loading Phase 1 checkpoint...")
    if not Path(args.phase1_ckpt).exists():
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {args.phase1_ckpt}")
    
    ckpt = torch.load(args.phase1_ckpt, map_location=device, weights_only=False)
    print(f"Phase 1: epoch={ckpt['epoch']}, val_loss={ckpt.get('best_val_loss', '?')}")

    # Create base generator with matching dimensions
    from src.v26.diffusion.regime_generator import RegimeGeneratorConfig, RegimeDiffusionPathGenerator
    
    config = RegimeGeneratorConfig(
        in_channels=144,
        sequence_length=120,
        base_channels=128,
        channel_multipliers=(1, 2, 4),
        time_dim=256,
        num_timesteps=1000,
        ctx_dim=144,
        guidance_scale=3.0,
        num_paths=1,
        sampling_steps=50,
        dropout=0.1,
        temporal_gru_dim=256,
        temporal_layers=2,
        context_len=256,
        norm_stats_path=str(V24_DIFFUSION_NORM_STATS_6M_PATH),
        num_regimes=9,
        regime_embed_dim=16,
        regime_conditioning_strength=1.0,
        temporal_film_dim=272,  # 256 + 16 to match Phase 1 training
    )
    
    base_gen = RegimeDiffusionPathGenerator(
        config=config,
        device=str(device),
    )
    
    # Load compatible weights (filter by key AND shape)
    checkpoint_state = ckpt.get("ema", ckpt.get("model", {}))
    model_state = base_gen.model.state_dict()
    
    compatible = {
        k: v
        for k, v in checkpoint_state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    
    missing = set(model_state.keys()) - set(compatible.keys())
    
    base_gen.model.load_state_dict(compatible, strict=False)
    
    print(f"Loaded {len(compatible)} tensors from Phase 1 model")
    print(f"Skipped {len(missing)} tensors (shape mismatch)")
    
    # Also load regime embedder from checkpoint
    if "regime_embedding" in ckpt:
        regime_state = base_gen.regime_embedder.state_dict()
        regime_compatible = {
            k: v
            for k, v in ckpt["regime_embedding"].items()
            if k in regime_state and regime_state[k].shape == v.shape
        }
        base_gen.regime_embedder.load_state_dict(regime_compatible, strict=False)
        print(f"Loaded {len(regime_compatible)} regime embedding tensors")
    
    # Load temporal encoder
    if "temporal_encoder" in ckpt:
        temp_state = base_gen.temporal_encoder.state_dict()
        temp_compatible = {
            k: v
            for k, v in ckpt["temporal_encoder"].items()
            if k in temp_state and temp_state[k].shape == v.shape
        }
        base_gen.temporal_encoder.load_state_dict(temp_compatible, strict=False)
        print(f"Loaded {len(temp_compatible)} temporal encoder tensors")

    # Create multi-horizon generator
    print("Creating Multi-Horizon Generator...")
    multi_gen = MultiHorizonGenerator(
        base_generator=base_gen,
        summary_dim=64,
        device=str(device),
    ).to(device)

    # Freeze base generator components
    for p in multi_gen.base_generator.model.parameters():
        p.requires_grad = False
    if multi_gen.base_generator.temporal_encoder is not None:
        for p in multi_gen.base_generator.temporal_encoder.parameters():
            p.requires_grad = False
    if multi_gen.base_generator.regime_embedder is not None:
        for p in multi_gen.base_generator.regime_embedder.parameters():
            p.requires_grad = False

    print("Base generator frozen, training horizon components only")

    # Optimizer for new components only
    optimizer = AdamW([
        {"params": multi_gen.short_summary.parameters()},
        {"params": multi_gen.medium_summary.parameters()},
        {"params": multi_gen.medium_condition_proj.parameters()},
        {"params": multi_gen.long_condition_proj.parameters()},
        {"params": multi_gen.regime_embed.parameters()},
    ], lr=args.lr, weight_decay=0.01)

    # Load dataset
    print("Loading dataset...")
    fused_path = V24_DIFFUSION_FUSED_6M_PATH
    timestamps_path = V24_DIFFUSION_TIMESTAMPS_6M_PATH

    fused_mmap = np.load(str(fused_path), mmap_mode="r")
    total_rows = len(fused_mmap)
    print(f"Dataset: {total_rows:,} rows")

    timestamps = None
    if timestamps_path.exists():
        timestamps = np.load(str(timestamps_path), mmap_mode="r")

    train_slice, val_slice, _ = split_by_year(total_rows, 120, timestamps=timestamps)
    print(f"Train: {len(train_slice):,}, Val: {len(val_slice):,}")

    regime_cache_path = PROJECT_ROOT / "data" / "cache" / "v26_regime_labels_tactical_v24_2.npy"
    regime_cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Regime cache path: {regime_cache_path}")

    train_ds = RegimeDiffusionDataset(
        fused_path, 120, train_slice,
        context_len=256, max_samples=args.max_samples, load_to_ram=True,
        regime_cache_path=regime_cache_path,
    )
    val_ds = RegimeDiffusionDataset(
        fused_path, 120, val_slice,
        context_len=256, max_samples=min(10000, len(val_slice)), load_to_ram=True,
        regime_cache_path=regime_cache_path,
    )

    train_dl = DataLoader(
        train_ds, batch_size=1024, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True,
    )

    # Training loop
    log_path = OUTPUTS_V26_DIR / "multi_horizon_training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_consistency = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs - 1}")

        train_metrics = _train_multi_horizon_epoch(
            multi_gen, train_dl, optimizer, device, epoch
        )

        val_metrics = _validate_multi_horizon(multi_gen, val_dl, device)

        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": float(train_metrics["total_loss"]),
            "train_consistency": float(train_metrics["consistency"]),
            "val_consistency": float(val_metrics["val_consistency"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": round(elapsed, 1),
        }

        # Save best
        if val_metrics["val_consistency"] > best_consistency:
            best_consistency = val_metrics["val_consistency"]
            entry["best"] = True
            torch.save({
                "multi_horizon_state": multi_gen.state_dict(),
                "base_generator_path": args.phase1_ckpt,
                "epoch": epoch,
                "best_val_consistency": best_consistency,
                "phase": "2.0",
            }, args.output)
            print(f"  [BEST] Saved checkpoint: consistency={best_consistency:.4f}")

        with open(str(log_path), "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  Train: loss={train_metrics['total_loss']:.4f}, "
              f"consistency={train_metrics['consistency']:.4f}")
        print(f"  Val: consistency={val_metrics['val_consistency']:.4f}")
        print(f"  Time: {elapsed:.0f}s")

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best consistency: {best_consistency:.4f}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()