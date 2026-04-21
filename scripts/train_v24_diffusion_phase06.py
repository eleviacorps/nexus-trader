"""Phase 0.6 — Temporal Realism Fine-Tuning

Continues from epoch-29 checkpoint with explicit realism losses.
Reduces guidance scale to mitigate overdispersion.
"""

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
    V24_DIFFUSION_TIMESTAMPS_6M_PATH,
    OUTPUTS_V24_DIR,
)
from src.v24.diffusion.dataset import DiffusionDataset, split_by_year
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


def _train_one_epoch(model, temporal_encoder, scheduler, dataloader, dataset, optimizer, device, use_amp, acf_w=0.10, vol_w=0.10, std_w=0.05):
    model.train()
    temporal_encoder.train()
    dataset.new_epoch()
    total_loss = total_diff = total_acf = total_vol = total_std = 0.0
    n_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (window, past_ctx) in enumerate(dataloader):
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            temporal_seq, temporal_emb, _ = temporal_encoder(past_ctx)
            context = past_ctx[:, -1, :]

            loss_dict = scheduler.training_loss_with_realism(
                model, window, context,
                temporal_seq=temporal_seq, temporal_emb=temporal_emb,
                acf_weight=acf_w, vol_weight=vol_w, std_weight=std_w,
            )
            loss = loss_dict["total"]

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(temporal_encoder.parameters()), 1.0)
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
            print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}, diff={loss_dict['diffusion'].item():.4f}, "
                  f"acf={loss_dict['acf'].item():.4f}, vol={loss_dict['vol'].item():.4f}, std={loss_dict['std'].item():.4f}")

    return {
        "total": total_loss / max(n_batches, 1),
        "diffusion": total_diff / max(n_batches, 1),
        "acf": total_acf / max(n_batches, 1),
        "vol": total_vol / max(n_batches, 1),
        "std": total_std / max(n_batches, 1),
    }


@torch.no_grad()
def _validate(model, temporal_encoder, scheduler, dataloader, device, use_amp):
    model.eval()
    temporal_encoder.eval()
    total = 0.0
    n = 0
    for window, past_ctx in dataloader:
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            temporal_seq, temporal_emb, _ = temporal_encoder(past_ctx)
            context = past_ctx[:, -1, :]
            loss = scheduler.training_loss(model, window, context, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Phase 0.6 — Realism Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--acf-weight", type=float, default=0.10)
    parser.add_argument("--vol-weight", type=float, default=0.10)
    parser.add_argument("--std-weight", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=100000)
    args = parser.parse_args()

    _set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Device: {device}")
    print(f"Phase 0.6 Fine-Tuning: acf_w={args.acf_weight}, vol_w={args.vol_weight}, std_w={args.std_weight}")
    print(f"LR: {args.lr}, epochs: {args.epochs}")

    # Load best checkpoint
    ckpt_path = V24_DIFFUSION_CHECKPOINT_6M_PATH
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    print(f"Resuming from epoch {ckpt['epoch']}, val_loss={ckpt.get('best_val_loss', '?')}")

    fused_path = V24_DIFFUSION_FUSED_6M_PATH
    timestamps_path = V24_DIFFUSION_TIMESTAMPS_6M_PATH
    out_ckpt_path = ckpt_path.parent / "diffusion_unet1d_v2_6m_phase06.pt"

    timestamps = None
    if timestamps_path.exists():
        timestamps = np.load(str(timestamps_path), mmap_mode="r")

    fused_mmap = np.load(str(fused_path), mmap_mode="r")
    total = len(fused_mmap)
    print(f"Dataset: {total} rows")

    train_slice, val_slice, _ = split_by_year(total, 120, timestamps=timestamps)
    print(f"Train: {len(train_slice)}, Val: {len(val_slice)}")

    train_ds = DiffusionDataset(fused_path, 120, train_slice, context_len=256, max_samples=args.max_samples, load_to_ram=True)
    val_ds = DiffusionDataset(fused_path, 120, val_slice, context_len=256, max_samples=min(50000, len(val_slice)), load_to_ram=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DiffusionUNet1D(in_channels=144, base_channels=128, channel_multipliers=(1, 2, 4),
                          time_dim=256, num_res_blocks=2, ctx_dim=144, temporal_dim=256, d_gru=256).to(device)
    model.load_state_dict(ckpt["ema"])
    print(f"Model loaded (EMA)")

    temporal_encoder = TemporalEncoder(in_features=144, d_gru=256, num_layers=2, film_dim=256).to(device)
    temporal_encoder.load_state_dict(ckpt.get("temporal_encoder", {}))
    temporal_encoder.train()

    scheduler = NoiseScheduler(1000).to(device)

    optimizer = AdamW(list(model.parameters()) + list(temporal_encoder.parameters()), lr=args.lr, weight_decay=0.01)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = EMA(model, decay=0.999)

    log_path = OUTPUTS_V24_DIR / "diffusion_v2_phase06_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = ckpt["epoch"] + 1

    print(f"\nFine-tuning from epoch {start_epoch} to {start_epoch + args.epochs - 1}")
    print(f"Log: {log_path}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        loss_dict = _train_one_epoch(model, temporal_encoder, scheduler, train_dl, train_ds,
                                   optimizer, device, use_amp, args.acf_weight, args.vol_weight, args.std_weight)
        ema.update(model)
        elapsed = time.time() - t0

        entry = {"epoch": epoch, "train_total": loss_dict["total"], "train_diff": loss_dict["diffusion"],
                "train_acf": loss_dict["acf"], "train_vol": loss_dict["vol"], "train_std": loss_dict["std"],
                "lr": optimizer.param_groups[0]["lr"], "elapsed_s": round(elapsed, 1)}

        if (epoch + 1) % 1 == 0:
            ema.apply(model)
            val_loss = _validate(model, temporal_encoder, scheduler, val_dl, device, use_amp)
            ema.restore(model)
            entry["val_loss"] = val_loss
            is_best = val_loss < ckpt.get("best_val_loss", float("inf"))
            if is_best:
                entry["best"] = True
                torch.save({
                    "model": model.state_dict(),
                    "temporal_encoder": temporal_encoder.state_dict(),
                    "ema": ema.shadow,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": val_loss,
                    "phase": "0.6",
                }, str(out_ckpt_path))
                print(f"  **SAVED BEST** val_loss={val_loss:.6f}")

        lr_scheduler.step()
        with open(str(log_path), "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"Epoch {epoch}: total={loss_dict['total']:.4f}, diff={loss_dict['diffusion']:.4f}, "
              f"acf={loss_dict['acf']:.4f}, vol={loss_dict['vol']:.4f}, std={loss_dict['std']:.4f} | {elapsed:.0f}s")

    print(f"\nFine-tuning complete. Best model: {out_ckpt_path}")


if __name__ == "__main__":
    main()