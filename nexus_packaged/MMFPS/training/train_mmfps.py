"""Training pipeline for MMFPS.

Trains generator + selector jointly with:
- Generator: diversity loss + regime consistency
- Selector: weighted return MSE only (no path labels)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

nexus_path = Path(__file__).resolve().parents[2]
if str(nexus_path) not in sys.path:
    sys.path.insert(0, str(nexus_path))
from MMFPS.mmfps import MMFPS
from MMFPS.selector.distribution_selector import SelectorLoss
from MMFPS.generator.path_generator import DiversityLoss


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        logger.addHandler(logging.FileHandler(log_file))
    return logger


class MarketDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        start_idx: int = 0,
        num_samples: int | None = None,
    ):
        self.features = features
        self.close_prices = close_prices
        self.start_idx = start_idx

        max_idx = min(len(features) - 20, len(close_prices) - 20)
        if num_samples is not None:
            max_idx = min(max_idx, start_idx + num_samples)
        self.end_idx = max_idx

    def __len__(self) -> int:
        return max(0, self.end_idx - self.start_idx)

    def __getitem__(self, idx: int):
        p = self.start_idx + idx
        context = self.features[p].astype(np.float32)
        actual_end = p + 21
        if actual_end > len(self.close_prices):
            actual_end = len(self.close_prices)
        actual = self.close_prices[p:actual_end].astype(np.float32)
        if len(actual) < 21:
            pad = np.zeros(21, dtype=np.float32)
            pad[: len(actual)] = actual
            actual = pad
        return {"context": context, "actual": actual}


def load_data(num_samples: int = 600000):
    logger = logging.getLogger("mmfps.train")
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]

    features_path = repo_root / "data/features/diffusion_fused_6m.npy"
    ohlcv_path = repo_root / "nexus_packaged/data/ohlcv.parquet"

    features = np.load(str(features_path), mmap_mode="r")[:num_samples]
    logger.info(f"Features: {features.shape}")

    ohlcv_df = pd.read_parquet(str(ohlcv_path))
    close_prices = ohlcv_df["close"].values.astype(np.float32)
    logger.info(f"Close prices: {close_prices.shape}")

    return features, close_prices


def train_epoch(model, loader, gen_loss_fn, sel_loss_fn, optimizer, device, epoch, logger):
    model.train()
    total = {
        k: 0.0 for k in [
            "loss", "gen_loss", "sel_loss", "mse_loss", "bce_loss",
            "diversity_loss", "entropy", "corr_with_actual", "dir_accuracy",
            "effective_paths", "weight_std", "path_diversity",
        ]
    }
    n = 0

    for batch in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        context = batch["context"].to(device)
        actual = batch["actual"].to(device)

        mmfps_out = model(context)
        sel_output = model.selector(context, mmfps_out.paths)
        sel_metrics = sel_loss_fn(sel_output, actual)

        div_metrics = gen_loss_fn(mmfps_out.paths)

        loss = sel_metrics["loss"] + div_metrics["diversity_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in [
            "loss", "mse_loss", "bce_loss", "diversity_loss", "entropy",
            "corr_with_actual", "dir_accuracy", "effective_paths", "weight_std",
        ]:
            if k in sel_metrics:
                total[k] += sel_metrics[k].item()
        total["gen_loss"] += div_metrics["diversity_loss"].item()
        total["path_diversity"] += div_metrics["ret_std"].item()
        n += 1

    for k in total:
        total[k] /= n

    logger.info(
        f"T {epoch} | loss={total['loss']:.6f} | gen={total['gen_loss']:.4f} | "
        f"mse={total['mse_loss']:.6f} | bce={total['bce_loss']:.4f} | "
        f"div={total['path_diversity']:.4f} | "
        f"entropy={total['entropy']:.4f} | eff_paths={total['effective_paths']:.1f} | "
        f"corr={total['corr_with_actual']:.4f} | dir={total['dir_accuracy']:.4f}"
    )
    return total


@torch.no_grad()
def validate(model, loader, sel_loss_fn, device, epoch, logger):
    model.eval()
    total = {
        k: 0.0 for k in [
            "loss", "mse_loss", "bce_loss", "entropy",
            "corr_with_actual", "dir_accuracy", "effective_paths",
            "weight_std", "calib_error",
        ]
    }
    n = 0

    for batch in tqdm(loader, desc=f"Val {epoch}", leave=False):
        context = batch["context"].to(device)
        actual = batch["actual"].to(device)

        mmfps_out = model(context)
        sel_output = model.selector(context, mmfps_out.paths)
        metrics = sel_loss_fn(sel_output, actual)

        for k in total:
            if k in metrics:
                total[k] += metrics[k].item()
        total["loss"] += metrics["loss"].item()
        n += 1

    for k in total:
        total[k] /= n

    logger.info(
        f"V {epoch} | loss={total['loss']:.6f} | mse={total['mse_loss']:.6f} | "
        f"bce={total['bce_loss']:.4f} | entropy={total['entropy']:.4f} | "
        f"eff_paths={total['effective_paths']:.1f} | w_std={total['weight_std']:.4f} | "
        f"corr={total['corr_with_actual']:.4f} | dir={total['dir_accuracy']:.4f}"
    )
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=600000)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--num-paths", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path("nexus_packaged/MMFPS/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    logger = setup_logger("mmfps.train", str(log_file))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {args}")

    features, close_prices = load_data(args.num_samples)

    total_samples = min(len(features) - 128, args.num_samples)
    train_end = int(total_samples * args.train_split)

    train_dataset = MarketDataset(features, close_prices, start_idx=0, num_samples=train_end)
    val_dataset = MarketDataset(features, close_prices, start_idx=train_end, num_samples=total_samples - train_end)

    logger.info(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = MMFPS({
        "num_paths": args.num_paths,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
    }).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    sel_loss_fn = SelectorLoss(mse_weight=1.0, bce_weight=0.5, entropy_weight=0.05)
    gen_loss_fn = DiversityLoss(margin=0.05, weight=1.0)

    best_corr = -1.0
    best_dir = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, gen_loss_fn, sel_loss_fn, optimizer, device, epoch, logger)
        val_metrics = validate(model, val_loader, sel_loss_fn, device, epoch, logger)
        scheduler.step()
        epoch_time = time.time() - t0

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "time": epoch_time})

        if val_metrics["corr_with_actual"] > best_corr:
            best_corr = val_metrics["corr_with_actual"]
            best_dir = val_metrics["dir_accuracy"]
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, output_dir / "best_mmfps.pt")
            logger.info(f"*** Saved best (corr={best_corr:.4f}, dir={best_dir:.4f})")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()}, output_dir / "final_mmfps.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete. Best val corr: {best_corr:.4f}, dir: {best_dir:.4f}")

    result = {
        "version": "mmfps",
        "best_corr": best_corr,
        "best_dir": best_dir,
        "final_history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()