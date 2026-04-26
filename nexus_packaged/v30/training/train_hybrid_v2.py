"""V30.5 Hybrid Selector v2 - Real signal comparison.

Compares:
  1. DistributionSelector (baseline)
  2. HybridSelector (fake signals - v1)
  3. HybridSelectorV2 (real signals - v2)

REPLACEMENT CONDITION: val_corr >= 0.30 AND dir_acc >= 0.61
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
from v30.models.selector.hybrid_selector_v2 import HybridSelectorV2, HybridV2Loss


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
    logger = logging.getLogger("v30_5_v2.train")
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


def validate_model(model, loader, loss_fn, device, model_name: str):
    model.eval()
    total = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            context = batch["context"].to(device)
            paths = batch["paths"].to(device)
            actual = batch["actual"].to(device)

            if isinstance(model, (HybridSelector, HybridSelectorV2)):
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
    logger = setup_logger("v30_5_v2.train", str(output_dir / "hybrid_v2_training.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    features, paths, close_prices = load_data()

    train_split = args.num_samples
    train_dataset = HybridDataset(features, paths, close_prices, args.num_samples, 0)
    val_dataset = HybridDataset(features, paths, close_prices, args.val_samples, train_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    dist_model = DistributionSelector(feature_dim=144, path_dim=20, num_paths=64, hidden_dim=128, dropout=0.1).to(device)
    v2_model = HybridSelectorV2(feature_dim=144, path_dim=20, num_paths=64, hidden_dim=256, dropout=0.1).to(device)

    dist_opt = optim.Adam(dist_model.parameters(), lr=args.lr, weight_decay=1e-5)
    v2_opt = optim.Adam(v2_model.parameters(), lr=args.lr, weight_decay=1e-5)

    dist_scheduler = optim.lr_scheduler.CosineAnnealingLR(dist_opt, T_max=args.epochs)
    v2_scheduler = optim.lr_scheduler.CosineAnnealingLR(v2_opt, T_max=args.epochs)

    dist_loss_fn = DistributionLoss(mse_weight=1.0, bce_weight=0.5, entropy_weight=0.01)
    v2_loss_fn = HybridV2Loss(mse_weight=1.0, bce_weight=0.5, calib_weight=0.1)

    best_dist_corr = -1.0
    best_v2_corr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")

        dist_model.train()
        v2_model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            context = batch["context"].to(device)
            batch_paths = batch["paths"].to(device)
            actual = batch["actual"].to(device)

            dist_opt.zero_grad()
            dist_out = dist_model(context, batch_paths)
            dist_metrics = dist_loss_fn(dist_out, batch_paths, actual)
            dist_metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(dist_model.parameters(), 1.0)
            dist_opt.step()

            v2_opt.zero_grad()
            v2_out = v2_model(context, batch_paths)
            v2_metrics = v2_loss_fn(v2_out, actual)
            v2_metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(v2_model.parameters(), 1.0)
            v2_opt.step()

        dist_scheduler.step()
        v2_scheduler.step()

        dist_val = validate_model(dist_model, val_loader, dist_loss_fn, device, "DIST")
        v2_val = validate_model(v2_model, val_loader, v2_loss_fn, device, "HYBRID_V2")

        def g(h, k): return h.get(k, 0.0)
        dist_corr = g(dist_val, "corr_with_actual")
        v2_corr = g(v2_val, "corr_with_actual")

        logger.info(f"  DIST     val: corr={dist_corr:.4f} dir_acc={g(dist_val,'dir_accuracy'):.4f} loss={g(dist_val,'loss'):.6f}")
        logger.info(f"  V2       val: corr={v2_corr:.4f} dir_acc={g(v2_val,'dir_accuracy'):.4f} loss={g(v2_val,'loss'):.6f}")

        if dist_corr > best_dist_corr:
            best_dist_corr = dist_corr
            torch.save({"epoch": epoch, "model_state_dict": dist_model.state_dict(), "val": dist_val}, output_dir / "best_distribution_selector.pt")
            logger.info(f"  [SAVED] best dist (corr={best_dist_corr:.4f})")

        if v2_corr > best_v2_corr:
            best_v2_corr = v2_corr
            torch.save({"epoch": epoch, "model_state_dict": v2_model.state_dict(), "val": v2_val}, output_dir / "best_hybrid_v2_selector.pt")
            logger.info(f"  [SAVED] best v2 (corr={best_v2_corr:.4f})")

        history.append({"epoch": epoch, "dist": dist_val, "v2": v2_val})

    torch.save({"epoch": args.epochs, "model_state_dict": v2_model.state_dict()}, output_dir / "final_hybrid_v2_selector.pt")

    final_dist = history[-1]["dist"]
    final_v2 = history[-1]["v2"]

    def gc(h): return max(g(history[i]["dist"], "corr_with_actual") for i in range(len(history)))
    def gv(h): return max(g(history[i]["v2"], "corr_with_actual") for i in range(len(history)))

    best_d = gc(history)
    best_v = gv(history)
    corr_imp = (best_v - best_d) / (best_d + 1e-6)

    final_dir_d = g(final_dist, "dir_accuracy")
    final_dir_v = g(final_v2, "dir_accuracy")
    dir_imp = final_dir_v - final_dir_d

    adopt = corr_imp >= 0.10 and dir_imp > 0

    summary = {
        "best_dist_corr": best_d,
        "best_v2_corr": best_v,
        "dist_dir_acc": final_dir_d,
        "v2_dir_acc": final_dir_v,
        "corr_improvement_pct": corr_imp * 100,
        "dir_improvement": dir_imp,
        "REPLACE_CONDITION": {
            "corr_improvement_ge_10pct": corr_imp >= 0.10,
            "dir_accuracy_improved": dir_imp > 0,
        },
        "VERDICT": "ADOPT HYBRID_V2" if adopt else "KEEP DISTRIBUTION_SELECTOR",
    }

    with open(output_dir / "hybrid_v2_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "hybrid_v2_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n========== COMPARISON SUMMARY ==========")
    logger.info(f"  DIST      best_corr={best_d:.4f}  final_dir={final_dir_d:.4f}")
    logger.info(f"  HYBRID_V2 best_corr={best_v:.4f}  final_dir={final_dir_v:.4f}")
    logger.info(f"  Corr improvement: {corr_imp*100:+.2f}%")
    logger.info(f"  Dir improvement:  {dir_imp:+.4f}")
    logger.info(f"  VERDICT: {summary['VERDICT']}")
    logger.info(f"=========================================")


if __name__ == "__main__":
    main()