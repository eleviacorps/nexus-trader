"""V30 Selector Training - unified model with distribution learning.

Following executive master prompt:
- Single unified model (NOT ensemble)
- Distribution-based reasoning (NOT argmax)
- reward = similarity(path, actual_future) as ground truth
- Loss: -sum(weights * normalized_rewards)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

# Add paths
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))


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


class V30Dataset(Dataset):
    """Dataset for V30 selector training.

    Loads:
        - features: [N, 144] market context
        - paths: [N, num_paths, horizon] diffusion paths
        - actual: [N, horizon] actual future prices

    Samples at 15-min intervals to match precomputed paths.
    """

    def __init__(
        self,
        features: np.ndarray,
        paths: np.ndarray,
        close_prices: np.ndarray,
        lookback: int = 128,
        horizon: int = 20,
        num_paths: int = 64,
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        self.features = features  # [N, 144]
        self.paths = paths  # [N_samples, num_paths, horizon]
        self.close_prices = close_prices  # [N]
        self.lookback = lookback
        self.horizon = horizon
        self.num_paths = num_paths

        if end_idx is None:
            end_idx = len(paths)

        self.start_idx = start_idx
        self.end_idx = min(end_idx, len(paths))

    def __len__(self) -> int:
        return max(0, self.end_idx - self.start_idx)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path_idx = self.start_idx + idx

        # Paths
        paths = self.paths[path_idx].astype(np.float32)

        # Context: last 'lookback' 15-min features before this point
        ctx_start = path_idx - self.lookback
        if ctx_start < 0:
            available = path_idx
            ctx_indices = [15 * i for i in range(available)]
            ctx = self.features[ctx_indices].astype(np.float32) if available > 0 else np.zeros((0, self.features.shape[1]), dtype=np.float32)
            pad_size = self.lookback - available
            context = np.vstack([np.zeros((pad_size, self.features.shape[1]), dtype=np.float32), ctx])
        else:
            ctx_indices = [15 * i for i in range(ctx_start, path_idx)]
            context = self.features[ctx_indices].astype(np.float32)

        # Actual: OHLCV close prices at 15-min intervals
        actual_start = path_idx * 15
        actual_indices = [actual_start + i * 15 for i in range(self.horizon + 1)]
        actual = np.array([self.close_prices[min(i, len(self.close_prices) - 1)] for i in actual_indices], dtype=np.float32)

        return {
            "context": context,
            "paths": paths,
            "actual": actual,
        }


def load_data(config: dict):
    """Load features, OHLCV, and precomputed paths."""
    from v30.training.dataloader import load_alignment_data

    logger = logging.getLogger("v30.train")

    # Load features and OHLCV
    features, ohlcv_df = load_alignment_data(config.get("data", {}))
    logger.info(f"Loaded features: {features.shape}")

    # Convert to float32
    if features.dtype != np.float32:
        features = features.astype(np.float32)
        logger.info(f"Converted features to float32")

    # Extract close prices
    if isinstance(ohlcv_df, pd.DataFrame):
        close_prices = ohlcv_df["close"].values.astype(np.float32)
    else:
        close_prices = np.array([row["close"] for row in ohlcv_df], dtype=np.float32)

    logger.info(f"Close prices: {close_prices.shape}")

    # Load precomputed paths
    paths_file = config["data"].get("precomputed_paths")
    if paths_file:
        paths_file = Path(paths_file)
        if paths_file.exists():
            paths = np.load(paths_file)
            if paths.dtype != np.float32:
                paths = paths.astype(np.float32)
            logger.info(f"Loaded precomputed paths: {paths.shape}")
        else:
            raise FileNotFoundError(f"Precomputed paths not found: {paths_file}")
    else:
        raise ValueError("precomputed_paths not in config")

    return features, paths, close_prices


def train_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    device,
    epoch: int,
    logger,
) -> dict[str, float]:
    """Train one epoch."""
    model.train()

    total_metrics = {
        "loss": 0.0,
        "expected_reward": 0.0,
        "entropy": 0.0,
        "corr_with_actual": 0.0,
        "calib_error": 0.0,
        "max_weight": 0.0,
        "weight_std": 0.0,
        "reward_range": 0.0,
    }

    num_batches = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        context = batch["context"].to(device)
        paths = batch["paths"].to(device)
        actual = batch["actual"].to(device)

        # Forward
        output = model(context, paths)
        metrics = loss_fn(output, paths, actual)

        # Backward
        optimizer.zero_grad()
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate (only keys in total_metrics)
        for k in total_metrics:
            if k in metrics:
                total_metrics[k] += metrics[k].item()
        total_metrics["loss"] += metrics["loss"].item()
        num_batches += 1

    # Average
    for k in total_metrics:
        total_metrics[k] /= num_batches

    logger.info(
        f"Epoch {epoch} | "
        f"loss: {total_metrics['loss']:.4f} | "
        f"reward: {total_metrics['expected_reward']:.4f} | "
        f"corr: {total_metrics['corr_with_actual']:.4f} | "
        f"max_w: {total_metrics['max_weight']:.3f}"
    )

    return total_metrics


def validate(
    model,
    val_loader,
    loss_fn,
    device,
    epoch: int,
    logger,
) -> dict[str, float]:
    """Validate."""
    model.eval()

    total_metrics = {
        "loss": 0.0,
        "expected_reward": 0.0,
        "entropy": 0.0,
        "corr_with_actual": 0.0,
        "calib_error": 0.0,
        "max_weight": 0.0,
        "reward_range": 0.0,
    }

    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val {epoch}"):
            context = batch["context"].to(device)
            paths = batch["paths"].to(device)
            actual = batch["actual"].to(device)

            output = model(context, paths)
            metrics = loss_fn(output, paths, actual)

            for k in total_metrics:
                if k in metrics:
                    total_metrics[k] += metrics[k].item()
            total_metrics["loss"] += metrics["loss"].item()
            num_batches += 1

    for k in total_metrics:
        total_metrics[k] /= num_batches

    logger.info(
        f"Val {epoch} | "
        f"loss: {total_metrics['loss']:.4f} | "
        f"corr: {total_metrics['corr_with_actual']:.4f} | "
        f"calib: {total_metrics['calib_error']:.4f}"
    )

    return total_metrics


def main():
    parser = argparse.ArgumentParser(description="Train V30 Unified Selector")
    parser.add_argument("--config", type=str,
                        default="nexus_packaged/v30/configs/v30_config.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str,
                        default="nexus_packaged/v30/models/selector")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "v30.train",
        log_file=str(output_dir / "training.log")
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    features, paths, close_prices = load_data(config)

    # Config
    lookback = config["model"]["generator"]["lookback"]
    horizon = config["model"]["generator"]["horizon"]
    num_paths = config["model"]["generator"]["num_paths"]
    feature_dim = config["model"]["feature_dim"]

    # Split based on number of path samples
    num_path_samples = len(paths)
    split_idx = int(num_path_samples * 0.8)
    logger.info(f"Train samples: 0-{split_idx}, Val samples: {split_idx}-{num_path_samples}")

    train_dataset = V30Dataset(
        features=features,
        paths=paths,
        close_prices=close_prices,
        lookback=lookback,
        horizon=horizon,
        num_paths=num_paths,
        start_idx=0,
        end_idx=split_idx,
    )

    val_dataset = V30Dataset(
        features=features,
        paths=paths,
        close_prices=close_prices,
        lookback=lookback,
        horizon=horizon,
        num_paths=num_paths,
        start_idx=split_idx,
        end_idx=num_path_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    from v30.models.selector.unified_selector import UnifiedSelector, V30TrainingLoss

    model = UnifiedSelector(
        feature_dim=feature_dim,
        path_dim=horizon,
        num_paths=num_paths,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = V30TrainingLoss(
        direction_weight=0.4,
        magnitude_weight=0.3,
        structure_weight=0.3,
        entropy_weight=0.05,
        reward_normalization=True,
    )

    # Train
    best_corr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, logger
        )

        val_metrics = validate(
            model, val_loader, loss_fn, device, epoch, logger
        )

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Save best by correlation
        if val_metrics["corr_with_actual"] > best_corr:
            best_corr = val_metrics["corr_with_actual"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, output_dir / "best_selector.pt")
            logger.info(f"Saved best model (corr={best_corr:.4f})")

        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }, output_dir / f"selector_epoch{epoch}.pt")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
    }, output_dir / "final_selector.pt")

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete. Best corr: {best_corr:.4f}")


if __name__ == "__main__":
    main()