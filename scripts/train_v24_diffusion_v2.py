from __future__ import annotations

import argparse
import json
import math
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
    V24_DIFFUSION_CHECKPOINT_PATH,
    V24_DIFFUSION_FUSED_PATH,
    V24_DIFFUSION_NORM_STATS_PATH,
    V24_DIFFUSION_TIMESTAMPS_PATH,
    OUTPUTS_V24_DIR,
)
from src.v24.diffusion.dataset import DiffusionDataset, DatasetSlice, split_by_year
from src.v24.diffusion.scheduler import NoiseScheduler
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
    scheduler: NoiseScheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (window, context) in enumerate(dataloader):
        window = window.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = scheduler.training_loss(model, window, context)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: DiffusionUNet1D,
    scheduler: NoiseScheduler,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for window, context in dataloader:
        window = window.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = scheduler.training_loss(model, window, context)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _save_checkpoint(
    model: DiffusionUNet1D,
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
    parser = argparse.ArgumentParser(description="Train V24 Diffusion U-Net v2 (epsilon-prediction)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}")

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
        "ema_decay": args.ema_decay,
        "seed": args.seed,
    }

    timestamps = None
    if V24_DIFFUSION_TIMESTAMPS_PATH.exists():
        timestamps = np.load(str(V24_DIFFUSION_TIMESTAMPS_PATH), mmap_mode="r")
        print(f"Timestamps loaded: {len(timestamps)} rows")

    fused = np.load(str(V24_DIFFUSION_FUSED_PATH), mmap_mode="r")
    total_rows = len(fused)
    print(f"Fused matrix: {fused.shape}")

    train_slice, val_slice, test_slice = split_by_year(
        total_rows, args.seq_len, timestamps=timestamps,
    )
    print(f"Train: {len(train_slice)}, Val: {len(val_slice)}, Test: {len(test_slice)}")

    train_ds = DiffusionDataset(V24_DIFFUSION_FUSED_PATH, args.seq_len, train_slice)
    val_ds = DiffusionDataset(V24_DIFFUSION_FUSED_PATH, args.seq_len, val_slice)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = DiffusionUNet1D(
        in_channels=100,
        base_channels=args.base_channels,
        channel_multipliers=(1, 2, 4),
        time_dim=args.time_dim,
        num_res_blocks=2,
        ctx_dim=100,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    noise_scheduler = NoiseScheduler(args.num_timesteps).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = EMA(model, decay=args.ema_decay)

    start_epoch = 0
    best_val_loss = float("inf")

    if V24_DIFFUSION_CHECKPOINT_PATH.exists():
        print(f"Resuming from {V24_DIFFUSION_CHECKPOINT_PATH}")
        ckpt = torch.load(str(V24_DIFFUSION_CHECKPOINT_PATH), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.shadow = ckpt.get("ema", ema.shadow)
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    log_path = OUTPUTS_V24_DIR / "diffusion_v2_training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training from epoch {start_epoch} to {args.epochs - 1}")
    print(f"Log: {log_path}")
    print(f"Checkpoint: {V24_DIFFUSION_CHECKPOINT_PATH}")
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = _train_one_epoch(model, noise_scheduler, train_dl, optimizer, device, args.grad_accum, use_amp)
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
            val_loss = _validate(model, noise_scheduler, val_dl, device, use_amp)
            ema.restore(model)
            entry["val_loss"] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, ema, optimizer, lr_scheduler, epoch, best_val_loss, V24_DIFFUSION_CHECKPOINT_PATH, train_config)
                entry["best"] = True

        if (epoch + 1) % args.save_interval == 0:
            _save_checkpoint(model, ema, optimizer, lr_scheduler, epoch, best_val_loss, V24_DIFFUSION_CHECKPOINT_PATH, train_config)

        lr_scheduler.step()

        with open(str(log_path), "a") as f:
            f.write(json.dumps(entry) + "\n")

        val_str = f" val={entry.get('val_loss', '?'):.4f}" if "val_loss" in entry else ""
        best_str = " **BEST**" if entry.get("best") else ""
        print(f"Epoch {epoch:3d} | train={train_loss:.4f}{val_str} | lr={entry['lr']:.2e} | {elapsed:.0f}s{best_str}")

    _save_checkpoint(model, ema, optimizer, lr_scheduler, args.epochs - 1, best_val_loss, V24_DIFFUSION_CHECKPOINT_PATH, train_config)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved: {V24_DIFFUSION_CHECKPOINT_PATH}")

    report = {
        "status": "complete",
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "params": n_params,
        "config": train_config,
    }
    report_path = OUTPUTS_V24_DIR / "diffusion_v2_training_report.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
