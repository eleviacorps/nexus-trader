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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    V24_DIFFUSION_CHECKPOINT_6M_PATH,
    V24_DIFFUSION_FUSED_6M_PATH,
    V24_DIFFUSION_NORM_STATS_6M_PATH,
    V24_DIFFUSION_TIMESTAMPS_6M_PATH,
    OUTPUTS_V24_DIR,
)
from src.v24.diffusion.dataset import DiffusionDataset, DatasetSlice, split_by_year
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.temporal_encoder import TemporalEncoder
from src.v24.diffusion.unet_1d import DiffusionUNet1D


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.detach()

    def apply(self, model: torch.nn.Module) -> None:
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.backup)
        self.backup = {}


def _train_one_epoch(
    model: DiffusionUNet1D,
    temporal_encoder: TemporalEncoder,
    scheduler: NoiseScheduler,
    dataloader: DataLoader,
    dataset: DiffusionDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
    use_amp: bool,
) -> float:
    model.train()
    temporal_encoder.train()
    dataset.new_epoch()
    total_loss = 0.0
    n_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (window, past_ctx) in enumerate(dataloader):
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            temporal_seq, temporal_emb, _ = temporal_encoder(past_ctx)
            context = past_ctx[:, -1, :]
            loss = scheduler.training_loss(model, window, context, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(temporal_encoder.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: DiffusionUNet1D,
    temporal_encoder: TemporalEncoder,
    scheduler: NoiseScheduler,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    temporal_encoder.eval()
    total_loss = 0.0
    n_batches = 0

    for window, past_ctx in dataloader:
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            temporal_seq, temporal_emb, _ = temporal_encoder(past_ctx)
            context = past_ctx[:, -1, :]
            loss = scheduler.training_loss(model, window, context, temporal_seq=temporal_seq, temporal_emb=temporal_emb)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _save_checkpoint(
    model: DiffusionUNet1D,
    temporal_encoder: TemporalEncoder,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler_lr: CosineAnnealingLR,
    epoch: int,
    best_val_loss: float,
    path: Path,
    train_config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "temporal_encoder": temporal_encoder.state_dict(),
            "ema": ema.shadow,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler_lr.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "train_config": train_config,
        },
        str(path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V24 Diffusion U-Net v2 (Phase 0.5 — temporal conditioning)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--in-channels", type=int, default=144)
    parser.add_argument("--ctx-dim", type=int, default=144)
    parser.add_argument("--temporal-gru-dim", type=int, default=256)
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=500000)
    parser.add_argument("--max-val-samples", type=int, default=50000)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}")
    print(f"In channels: {args.in_channels}, ctx_dim: {args.ctx_dim}")
    print(f"Temporal GRU dim: {args.temporal_gru_dim}, context_len: {args.context_len}")
    print(f"Max train samples: {args.max_train_samples}, Max val samples: {args.max_val_samples}")

    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seq_len": args.seq_len,
        "base_channels": args.base_channels,
        "time_dim": args.time_dim,
        "num_timesteps": args.num_timesteps,
        "in_channels": args.in_channels,
        "ctx_dim": args.ctx_dim,
        "temporal_gru_dim": args.temporal_gru_dim,
        "context_len": args.context_len,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
    }

    fused_path = V24_DIFFUSION_FUSED_6M_PATH
    timestamps_path = V24_DIFFUSION_TIMESTAMPS_6M_PATH
    checkpoint_path = V24_DIFFUSION_CHECKPOINT_6M_PATH

    timestamps = None
    if timestamps_path.exists():
        timestamps = np.load(str(timestamps_path), mmap_mode="r")
        print(f"Timestamps loaded: {len(timestamps)} rows")

    fused_mmap = np.load(str(fused_path), mmap_mode="r")
    total_rows = len(fused_mmap)
    print(f"Fused matrix shape: {fused_mmap.shape}")
    del fused_mmap

    train_slice, val_slice, test_slice = split_by_year(
        total_rows, args.seq_len, timestamps=timestamps,
    )
    print(f"Train slice: {len(train_slice):,}, Val slice: {len(val_slice):,}, Test slice: {len(test_slice):,}")

    print("Creating train dataset (RAM-loaded, subsampled) ...")
    train_ds = DiffusionDataset(
        fused_path, args.seq_len, train_slice,
        context_len=args.context_len,
        max_samples=args.max_train_samples,
        load_to_ram=True,
    )
    print("Creating val dataset (RAM-loaded, subsampled) ...")
    val_ds = DiffusionDataset(
        fused_path, args.seq_len, val_slice,
        context_len=args.context_len,
        max_samples=args.max_val_samples,
        load_to_ram=True,
    )

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    temporal_dim = args.time_dim if args.temporal_gru_dim > 0 else 0

    model = DiffusionUNet1D(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        channel_multipliers=(1, 2, 4),
        time_dim=args.time_dim,
        num_res_blocks=2,
        ctx_dim=args.ctx_dim,
        temporal_dim=temporal_dim,
        d_gru=args.temporal_gru_dim,
    ).to(device)

    temporal_encoder = TemporalEncoder(
        in_features=args.in_channels,
        d_gru=args.temporal_gru_dim,
        num_layers=2,
        film_dim=args.time_dim,
    ).to(device)

    n_params_unet = sum(p.numel() for p in model.parameters())
    n_params_te = sum(p.numel() for p in temporal_encoder.parameters())
    print(f"U-Net params: {n_params_unet:,}, Temporal encoder params: {n_params_te:,}")

    noise_scheduler = NoiseScheduler(args.num_timesteps).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(temporal_encoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = EMA(model, decay=args.ema_decay)

    start_epoch = 0
    best_val_loss = float("inf")

    if checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        temporal_encoder.load_state_dict(ckpt["temporal_encoder"])
        ema.shadow = ckpt.get("ema", ema.shadow)
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    log_path = OUTPUTS_V24_DIR / "diffusion_v2_6m_training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training from epoch {start_epoch} to {args.epochs - 1}")
    print(f"Training log: {log_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Train batches/epoch: ~{len(train_ds) // args.batch_size}")
    print(f"Val batches: ~{len(val_ds) // args.batch_size}")
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = _train_one_epoch(
            model, temporal_encoder, noise_scheduler, train_dl, train_ds,
            optimizer, device, args.grad_accum, use_amp,
        )
        ema.update(model)
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }

        if (epoch + 1) % args.val_interval == 0:
            ema.apply(model)
            val_loss = _validate(model, temporal_encoder, noise_scheduler, val_dl, device, use_amp)
            ema.restore(model)
            entry["val_loss"] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, temporal_encoder, ema, optimizer, lr_scheduler, epoch, best_val_loss, checkpoint_path, train_config)
                entry["best"] = True

        if (epoch + 1) % args.save_interval == 0:
            _save_checkpoint(model, temporal_encoder, ema, optimizer, lr_scheduler, epoch, best_val_loss, checkpoint_path, train_config)

        lr_scheduler.step()

        with open(str(log_path), "a") as f:
            f.write(json.dumps(entry) + "\n")

        val_str = f" val={entry.get('val_loss', '?'):.4f}" if "val_loss" in entry else ""
        best_str = " **BEST**" if entry.get("best") else ""
        print(f"Epoch {epoch:3d} | train={train_loss:.4f}{val_str} | lr={entry['lr']:.2e} | {elapsed:.0f}s{best_str}")

    _save_checkpoint(model, temporal_encoder, ema, optimizer, lr_scheduler, args.epochs - 1, best_val_loss, checkpoint_path, train_config)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")

    report = {
        "status": "complete",
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "unet_params": n_params_unet,
        "temporal_encoder_params": n_params_te,
        "config": train_config,
    }
    report_path = OUTPUTS_V24_DIR / "diffusion_v2_6m_training_report.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
