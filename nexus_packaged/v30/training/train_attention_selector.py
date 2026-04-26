"""Train AttentionPathSelector with Contrastive Path Loss on 1-minute data.

Forces context-path interaction via MultiheadAttention and teaches
the model which path matches reality via contrastive path selection loss.
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

nexus_path = Path(__file__).resolve().parents[2]
if str(nexus_path) not in sys.path:
    sys.path.insert(0, str(nexus_path))
from v30.models.selector.attention_path_selector import AttentionPathSelector, ContrastivePathLoss


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


class OneMinuteDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        paths: np.ndarray,
        close_prices: np.ndarray,
        start_idx: int = 0,
        num_samples: int | None = None,
    ):
        self.features = features
        self.paths = paths
        self.close_prices = close_prices
        self.start_idx = start_idx

        max_idx = min(len(paths), len(features) - 20)
        if num_samples is not None:
            max_idx = min(max_idx, start_idx + num_samples)
        self.end_idx = max_idx

    def __len__(self) -> int:
        return max(0, self.end_idx - self.start_idx)

    def __getitem__(self, idx: int):
        p = self.start_idx + idx
        context = self.features[p].astype(np.float32)
        paths = self.paths[p].astype(np.float32)
        actual_start = p
        actual_end = p + 21
        if actual_end > len(self.close_prices):
            actual_end = len(self.close_prices)
        actual = self.close_prices[actual_start:actual_end].astype(np.float32)
        if len(actual) < 21:
            pad = np.zeros(21, dtype=np.float32)
            pad[: len(actual)] = actual
            actual = pad
        return {"context": context, "paths": paths, "actual": actual}


def load_data(paths_file: str, num_samples: int = 600000):
    logger = logging.getLogger("v30.train_attn")
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]

    features_path = repo_root / "data/features/diffusion_fused_6m.npy"
    ohlcv_path = repo_root / "nexus_packaged/data/ohlcv.parquet"
    if Path(paths_file).is_absolute():
        paths_file_abs = Path(paths_file)
    else:
        paths_file_abs = repo_root / paths_file.lstrip("/")

    features = np.load(str(features_path), mmap_mode="r")[:num_samples]
    logger.info(f"Features: {features.shape}")

    ohlcv_df = pd.read_parquet(str(ohlcv_path))
    close_prices = ohlcv_df["close"].values.astype(np.float32)
    logger.info(f"Close prices: {close_prices.shape}")

    if not Path(paths_file_abs).exists():
        raise FileNotFoundError(f"Paths not found: {paths_file_abs}. Run precompute_1m_paths.py first.")

    lookback = 128
    horizon = 20
    actual_rows = min(len(features), num_samples)
    num_valid = min(actual_rows - lookback, num_samples - lookback) - lookback
    paths_shape = (num_valid, 64, horizon)
    paths_fp = np.memmap(str(paths_file_abs), dtype=np.float32, mode="r", shape=paths_shape)
    paths = np.array(paths_fp).astype(np.float32)
    logger.info(f"Paths: {paths.shape}")

    return features, paths, close_prices


def train_epoch(model, loader, loss_fn, optimizer, device, epoch, logger):
    model.train()
    total = {
        k: 0.0 for k in [
            "loss", "mse_loss", "bce_loss", "path_ce_loss", "entropy",
            "corr_with_actual", "dir_accuracy", "max_weight",
            "path_top1_acc", "path_top3_acc",
        ]
    }
    n = 0

    for batch in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        context = batch["context"].to(device)
        paths = batch["paths"].to(device)
        actual = batch["actual"].to(device)

        output = model(context, paths)
        metrics = loss_fn(output, paths, actual)

        optimizer.zero_grad()
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in total:
            if k in metrics:
                total[k] += metrics[k].item()
        total["loss"] += metrics["loss"].item()
        n += 1

    for k in total:
        total[k] /= n

    logger.info(
        f"T {epoch} | loss={total['loss']:.6f} | path_ce={total['path_ce_loss']:.4f} | "
        f"top1={total['path_top1_acc']:.4f} | top3={total['path_top3_acc']:.4f} | "
        f"corr={total['corr_with_actual']:.4f} | dir={total['dir_accuracy']:.4f}"
    )
    return total


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, logger):
    model.eval()
    total = {
        k: 0.0 for k in [
            "loss", "mse_loss", "bce_loss", "path_ce_loss", "entropy",
            "corr_with_actual", "dir_accuracy", "max_weight", "calib_error",
            "path_top1_acc", "path_top3_acc",
        ]
    }
    n = 0

    for batch in tqdm(loader, desc=f"Val {epoch}", leave=False):
        context = batch["context"].to(device)
        paths = batch["paths"].to(device)
        actual = batch["actual"].to(device)

        output = model(context, paths)
        metrics = loss_fn(output, paths, actual)

        for k in total:
            if k in metrics:
                total[k] += metrics[k].item()
        total["loss"] += metrics["loss"].item()
        n += 1

    for k in total:
        total[k] /= n

    logger.info(
        f"V {epoch} | loss={total['loss']:.6f} | path_ce={total['path_ce_loss']:.4f} | "
        f"top1={total['path_top1_acc']:.4f} | top3={total['path_top3_acc']:.4f} | "
        f"corr={total['corr_with_actual']:.4f} | dir={total['dir_accuracy']:.4f} | "
        f"calib={total['calib_error']:.4f}"
    )
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, default="nexus_packaged/v30/data/processed/v30_1m_paths.npy")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=600000)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--path-ce-weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path("nexus_packaged/v30/models/selector_attn")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("v30.train_attn", str(output_dir / "training.log"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    features, paths, close_prices = load_data(args.paths, args.num_samples)

    total_samples = min(len(paths), args.num_samples)
    train_end = int(total_samples * args.train_split)

    train_dataset = OneMinuteDataset(features, paths, close_prices, start_idx=0, num_samples=train_end)
    val_dataset = OneMinuteDataset(features, paths, close_prices, start_idx=train_end, num_samples=total_samples - train_end)

    logger.info(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = AttentionPathSelector(
        feature_dim=144,
        path_dim=20,
        num_paths=64,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dropout=0.1,
    ).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = ContrastivePathLoss(
        mse_weight=1.0,
        bce_weight=0.1,
        path_ce_weight=args.path_ce_weight,
        entropy_weight=0.01,
    )

    best_corr = -1.0
    best_dir = -1.0
    best_path_top1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, logger)
        val_metrics = validate(model, val_loader, loss_fn, device, epoch, logger)
        scheduler.step()
        epoch_time = time.time() - t0

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "time": epoch_time})

        if val_metrics["corr_with_actual"] > best_corr:
            best_corr = val_metrics["corr_with_actual"]
            best_dir = val_metrics["dir_accuracy"]
            best_path_top1 = val_metrics["path_top1_acc"]
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, output_dir / "best_attention_selector.pt")
            logger.info(f"*** Saved best (corr={best_corr:.4f}, dir={best_dir:.4f}, top1={best_path_top1:.4f})")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()}, output_dir / "final_attention_selector.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete. Best val corr: {best_corr:.4f}, dir: {best_dir:.4f}, path_top1: {best_path_top1:.4f}")

    result = {
        "version": "v30.attn_selector",
        "best_corr": best_corr,
        "best_dir": best_dir,
        "best_path_top1": best_path_top1,
        "final_history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()