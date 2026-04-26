"""V30.5 Hybrid Selector Training - Side-by-side comparison with distribution_selector.

Compare: Does adding regime + quant features improve over distribution_selector?
REPLACEMENT CONDITION: val_corr improves >= 10% AND dir_accuracy improves.
Otherwise: KEEP current distribution_selector.
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

import sys
nexus_path = Path(__file__).resolve().parents[2]
if str(nexus_path) not in sys.path:
    sys.path.insert(0, str(nexus_path))
from v30.models.selector.distribution_selector import DistributionSelector, DistributionLoss
from v30.models.selector.hybrid_selector import HybridSelector, HybridLoss


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


class HybridDataset(Dataset):
    def __init__(self, features, paths, close_prices, num_samples=50000, start_idx=0):
        self.features = features
        self.paths = paths
        self.close_prices = close_prices
        self.num_samples = num_samples
        self.start_idx = start_idx

    def __len__(self) -> int:
        return min(self.num_samples, len(self.paths) - self.start_idx)

    def __getitem__(self, idx: int):
        path_idx = self.start_idx + idx
        context = self.features[path_idx * 15].astype(np.float32)
        paths = self.paths[path_idx].astype(np.float32)
        actual_start = path_idx * 15
        actual_indices = [actual_start + i * 15 for i in range(21)]
        actual = np.array([self.close_prices[min(i, len(self.close_prices) - 1)] for i in actual_indices], dtype=np.float32)
        return {"context": context, "paths": paths, "actual": actual}


def load_data():
    logger = logging.getLogger("v30_5.train")
    project_root = Path(__file__).resolve().parents[3]

    features_path = project_root / "data/features/diffusion_fused_6m.npy"
    ohlcv_path = project_root / "nexus_packaged/data/ohlcv.parquet"
    paths_file = project_root / "nexus_packaged/v30/data/processed/v30_paths.npy"

    features = np.load(features_path).astype(np.float32)
    logger.info(f"Loaded features: {features.shape}")

    ohlcv_df = pd.read_parquet(ohlcv_path)
    close_prices = ohlcv_df["close"].values.astype(np.float32)
    logger.info(f"Close prices: {close_prices.shape}")

    paths = np.load(paths_file).astype(np.float32)
    logger.info(f"Loaded paths: {paths.shape}")

    return features, paths, close_prices


def train_epoch(model, loader, loss_fn, optimizer, device, epoch, logger, model_name="model"):
    model.train()
    total = {}
    n = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        context = batch["context"].to(device)
        paths = batch["paths"].to(device)
        actual = batch["actual"].to(device)

        if isinstance(model, HybridSelector):
            output = model(context, paths)
            metrics = loss_fn(output, actual)
        else:
            output = model(context, paths)
            metrics = loss_fn(output, paths, actual)

        optimizer.zero_grad()
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in metrics.items():
            if k not in total:
                total[k] = 0.0
            total[k] += v.item()
        n += 1

    for k in total:
        total[k] /= n

    logger.info(f"  {model_name} Epoch {epoch} | loss: {total.get('loss', 0):.6f} | mse: {total.get('mse_loss', total.get('corr_with_actual', 0)):.8f} | corr: {total.get('corr_with_actual', total.get('corr', 0)):.4f} | dir_acc: {total.get('dir_accuracy', total.get('dir_acc', 0)):.4f}")
    return total


def validate(model, loader, loss_fn, device, epoch, logger, model_name="model"):
    model.eval()
    total = {}
    n = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Val {epoch}", leave=False):
            context = batch["context"].to(device)
            paths = batch["paths"].to(device)
            actual = batch["actual"].to(device)

            if isinstance(model, HybridSelector):
                output = model(context, paths)
                metrics = loss_fn(output, actual)
            else:
                output = model(context, paths)
                metrics = loss_fn(output, paths, actual)

            for k, v in metrics.items():
                if k not in total:
                    total[k] = 0.0
                total[k] += v.item()
            n += 1

    for k in total:
        total[k] /= n

    logger.info(f"  {model_name} Val {epoch}   | loss: {total.get('loss', 0):.6f} | corr: {total.get('corr_with_actual', total.get('corr', 0)):.4f} | dir_acc: {total.get('dir_accuracy', total.get('dir_acc', 0)):.4f} | calib: {total.get('calib_error', 0):.4f}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    output_dir = Path("nexus_packaged/v30/models/selector")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("v30_5.train", str(output_dir / "hybrid_training.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    features, paths, close_prices = load_data()

    train_split = args.num_samples
    train_dataset = HybridDataset(features, paths, close_prices, args.num_samples, 0)
    val_dataset = HybridDataset(features, paths, close_prices, args.val_samples, train_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    hybrid_model = HybridSelector(feature_dim=144, path_dim=20, num_paths=64, hidden_dim=256, dropout=0.15).to(device)
    dist_model = DistributionSelector(feature_dim=144, path_dim=20, num_paths=64, hidden_dim=128, dropout=0.1).to(device)

    hybrid_opt = optim.Adam(hybrid_model.parameters(), lr=args.lr, weight_decay=1e-5)
    dist_opt = optim.Adam(dist_model.parameters(), lr=args.lr, weight_decay=1e-5)

    hybrid_scheduler = optim.lr_scheduler.CosineAnnealingLR(hybrid_opt, T_max=args.epochs)
    dist_scheduler = optim.lr_scheduler.CosineAnnealingLR(dist_opt, T_max=args.epochs)

    hybrid_loss_fn = HybridLoss(mse_weight=1.0, bce_weight=1.0, calib_weight=0.1)
    dist_loss_fn = DistributionLoss(mse_weight=1.0, bce_weight=0.5, entropy_weight=0.01)

    best_hybrid_corr = -1.0
    best_dist_corr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")

        logger.info("[Training]")
        train_h = train_epoch(hybrid_model, train_loader, hybrid_loss_fn, hybrid_opt, device, epoch, logger, "HYBRID")
        train_d = train_epoch(dist_model, train_loader, dist_loss_fn, dist_opt, device, epoch, logger, "DIST")

        logger.info("[Validation]")
        val_h = validate(hybrid_model, val_loader, hybrid_loss_fn, device, epoch, logger, "HYBRID")
        val_d = validate(dist_model, val_loader, dist_loss_fn, device, epoch, logger, "DIST")

        hybrid_scheduler.step()
        dist_scheduler.step()

        entry = {
            "epoch": epoch,
            "hybrid": {"train": train_h, "val": val_h},
            "dist": {"train": train_d, "val": val_d},
        }
        history.append(entry)

        def get_corr(h):
            return h.get("corr_with_actual", h.get("corr", 0.0))
        def get_dir(h):
            return h.get("dir_accuracy", h.get("dir_acc", 0.0))

        if get_corr(val_h) > best_hybrid_corr:
            best_hybrid_corr = get_corr(val_h)
            torch.save({"epoch": epoch, "model_state_dict": hybrid_model.state_dict(), "val": val_h}, output_dir / "best_hybrid_selector.pt")
            logger.info(f"  [SAVED] best hybrid (corr={best_hybrid_corr:.4f})")

        if get_corr(val_d) > best_dist_corr:
            best_dist_corr = get_corr(val_d)
            torch.save({"epoch": epoch, "model_state_dict": dist_model.state_dict(), "val": val_d}, output_dir / "best_distribution_selector.pt")
            logger.info(f"  [SAVED] best dist (corr={best_dist_corr:.4f})")

    torch.save({"epoch": args.epochs, "model_state_dict": hybrid_model.state_dict()}, output_dir / "final_hybrid_selector.pt")

    hybrid_val = history[-1]["hybrid"]["val"]
    dist_val = history[-1]["dist"]["val"]

    best_hybrid_final = max(get_corr(history[i]["hybrid"]["val"]) for i in range(len(history)))
    best_dist_final = max(get_corr(history[i]["dist"]["val"]) for i in range(len(history)))

    corr_improvement = (best_hybrid_final - best_dist_final) / (best_dist_final + 1e-6)
    dir_improvement = get_dir(hybrid_val) - get_dir(dist_val)

    summary = {
        "best_hybrid_corr": best_hybrid_final,
        "best_dist_corr": best_dist_final,
        "hybrid_dir_acc": get_dir(hybrid_val),
        "dist_dir_acc": get_dir(dist_val),
        "corr_improvement_pct": corr_improvement * 100,
        "dir_improvement": dir_improvement,
        "REPLACE_CONDITION": {
            "corr_improvement_ge_10pct": corr_improvement >= 0.10,
            "dir_accuracy_improved": dir_improvement > 0,
        },
        "VERDICT": "ADOPT HYBRID" if (corr_improvement >= 0.10 and dir_improvement > 0) else "KEEP DISTRIBUTION_SELECTOR",
    }

    with open(output_dir / "hybrid_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "hybrid_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n========== COMPARISON SUMMARY ==========")
    logger.info(f"  Hybrid  corr={best_hybrid_corr:.4f}  dir_acc={summary['hybrid_dir_acc']:.4f}")
    logger.info(f"  Dist    corr={best_dist_corr:.4f}  dir_acc={summary['dist_dir_acc']:.4f}")
    logger.info(f"  Corr improvement: {corr_improvement*100:.2f}%")
    logger.info(f"  Dir improvement:  {dir_improvement:+.4f}")
    logger.info(f"  VERDICT: {summary['VERDICT']}")
    logger.info(f"=========================================")


if __name__ == "__main__":
    main()