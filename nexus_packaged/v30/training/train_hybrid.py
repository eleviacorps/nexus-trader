"""V30 Hybrid Selector Training - State × Path → Distribution.

Following executive prompt:
- Hybrid architecture with fixed reward function
- Start small (50k samples)
- Validate correlation target > 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        logger.addHandler(logging.FileHandler(log_file))
    return logger


class HybridDataset(Dataset):
    """Dataset for hybrid selector training."""

    def __init__(
        self,
        features: np.ndarray,
        paths: np.ndarray,
        close_prices: np.ndarray,
        num_samples: int = 50000,
        start_idx: int = 0,
    ):
        self.features = features
        self.paths = paths
        self.close_prices = close_prices
        self.num_samples = num_samples
        self.start_idx = start_idx

    def __len__(self) -> int:
        return min(self.num_samples, len(self.paths) - self.start_idx)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path_idx = self.start_idx + idx
        context = self.features[path_idx * 15].astype(np.float32)
        paths = self.paths[path_idx].astype(np.float32)
        actual_start = path_idx * 15
        actual_indices = [actual_start + i * 15 for i in range(21)]
        actual = np.array([
            self.close_prices[min(i, len(self.close_prices) - 1)]
            for i in actual_indices
        ], dtype=np.float32)
        return {"context": context, "paths": paths, "actual": actual}


def load_data(config: dict):
    """Load features, OHLCV, and precomputed paths."""
    logger = logging.getLogger("v30.train")

    project_root = Path(__file__).resolve().parents[3]
    features_path = project_root / "data/features/diffusion_fused_6m.npy"
    ohlcv_path = project_root / "nexus_packaged/data/ohlcv.parquet"
    paths_file = project_root / "nexus_packaged/v30/data/processed/v30_paths.npy"

    features = np.load(features_path)
    if features.dtype != np.float32:
        features = features.astype(np.float32)
    logger.info(f"Loaded features: {features.shape}")

    ohlcv_df = pd.read_parquet(ohlcv_path)
    close_prices = ohlcv_df["close"].values.astype(np.float32)
    logger.info(f"Close prices: {close_prices.shape}")

    paths = np.load(paths_file)
    if paths.dtype != np.float32:
        paths = paths.astype(np.float32)
    logger.info(f"Loaded precomputed paths: {paths.shape}")

    return features, paths, close_prices


def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch: int, logger):
    model.train()
    total = {k: 0.0 for k in ["loss", "expected_reward", "contrastive", "corr_with_actual", "max_weight", "entropy", "reward_std_per_sample"]}
    num_batches = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
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
        num_batches += 1

    for k in total:
        total[k] /= num_batches

    logger.info(f"Epoch {epoch} | loss: {total['loss']:.4f} | corr: {total['corr_with_actual']:.4f} | max_w: {total['max_weight']:.3f} | reward_std: {total['reward_std_per_sample']:.4f}")
    return total


def validate(model, val_loader, loss_fn, device, epoch: int, logger):
    model.eval()
    total = {k: 0.0 for k in ["loss", "expected_reward", "contrastive", "corr_with_actual", "max_weight", "entropy", "calib_error", "reward_std_per_sample"]}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val {epoch}"):
            context = batch["context"].to(device)
            paths = batch["paths"].to(device)
            actual = batch["actual"].to(device)

            output = model(context, paths)
            metrics = loss_fn(output, paths, actual)

            for k in total:
                if k in metrics:
                    total[k] += metrics[k].item()
            total["loss"] += metrics["loss"].item()
            num_batches += 1

    for k in total:
        total[k] /= num_batches

    logger.info(f"Val {epoch} | loss: {total['loss']:.4f} | corr: {total['corr_with_actual']:.4f} | max_w: {total['max_weight']:.3f} | calib: {total['calib_error']:.4f}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Train V30 Hybrid Selector")
    parser.add_argument("--config", type=str, default="nexus_packaged/v30/configs/v30_config.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--val-samples", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="nexus_packaged/v30/models/selector")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("v30.train", log_file=str(output_dir / "training.log"))

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    features, paths, close_prices = load_data(config)

    train_split = args.num_samples
    val_split = train_split + args.val_samples
    logger.info(f"Train: 0-{train_split}, Val: {train_split}-{val_split}")

    train_dataset = HybridDataset(features, paths, close_prices, args.num_samples, 0)
    val_dataset = HybridDataset(features, paths, close_prices, args.val_samples, train_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Imports
    import sys
    nexus_path = Path(__file__).resolve().parents[2]
    if str(nexus_path) not in sys.path:
        sys.path.insert(0, str(nexus_path))
    from v30.models.selector.hybrid_selector import HybridSelector, HybridTrainingLoss

    model = HybridSelector(feature_dim=144, path_dim=20, num_paths=64, state_hidden=128, path_hidden=64, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = HybridTrainingLoss(w_time=0.3, w_direction=0.3, w_slope=0.2, w_volatility=0.2, contrastive_margin=0.5, contrastive_weight=1.0)

    best_corr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch, logger)
        val_metrics = validate(model, val_loader, loss_fn, device, epoch, logger)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["corr_with_actual"] > best_corr:
            best_corr = val_metrics["corr_with_actual"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics}, output_dir / "best_hybrid_selector.pt")
            logger.info(f"Saved best model (corr={best_corr:.4f})")

        if best_corr > 0.1:
            logger.info(f"\n✓ Target achieved: corr={best_corr:.4f} > 0.1")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()}, output_dir / "final_hybrid_selector.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nTraining complete. Best corr: {best_corr:.4f}")


if __name__ == "__main__":
    main()