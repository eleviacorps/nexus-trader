"""V30.7 Diffusion Selector Training - Compare with DistributionSelector.

HYPOTHESIS: Probabilistic diffusion modeling may capture return uncertainty better.
CONTROL: DistributionSelector (deterministic E[return])
ACCEPTANCE: val_corr > 0.30 AND dir_acc > 0.61

Architecture mirrors DistributionSelector:
    - Same state encoder, path encoder, attention, distribution encoder
    - Only difference: DDPM-style training with noise prediction objective
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
from v30.models.selector.diffusion_selector import DiffusionSelector, DiffusionLoss


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
    logger = logging.getLogger("v30_7.train")
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


def validate_model(model, loader, loss_fn, device, model_name: str, is_diffusion: bool = False):
    model.eval()
    total = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            context = batch["context"].to(device)
            paths = batch["paths"].to(device)
            actual = batch["actual"].to(device)

            if is_diffusion:
                output = model(context, paths, return_noised=False)
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    output_dir = Path("nexus_packaged/v30/models/selector")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("v30_7.train", str(output_dir / "diffusion_training.log"))

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
    diff_model = DiffusionSelector(
        feature_dim=144,
        path_dim=20,
        num_paths=64,
        hidden_dim=256,
        diffusion_steps=100,
        dropout=0.1,
    ).to(device)

    logger.info(f"DistributionSelector params: {sum(p.numel() for p in dist_model.parameters()):,}")
    logger.info(f"DiffusionSelector params: {sum(p.numel() for p in diff_model.parameters()):,}")

    dist_opt = optim.Adam(dist_model.parameters(), lr=args.lr, weight_decay=1e-5)
    diff_opt = optim.Adam(diff_model.parameters(), lr=args.lr, weight_decay=1e-5)

    dist_scheduler = optim.lr_scheduler.CosineAnnealingLR(dist_opt, T_max=args.epochs)
    diff_scheduler = optim.lr_scheduler.CosineAnnealingLR(diff_opt, T_max=args.epochs)

    dist_loss_fn = DistributionLoss(mse_weight=1.0, bce_weight=0.5, entropy_weight=0.01)
    diff_loss_fn = DiffusionLoss(
        noise_weight=1.0,
        bce_weight=0.5,
        entropy_weight=0.01,
        corr_reward_weight=0.1,
        dir_reward_weight=0.1,
        snr_reward_weight=0.05,
    )

    best_dist_corr = -1.0
    best_diff_corr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")

        dist_model.train()
        diff_model.train()

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

            diff_opt.zero_grad()
            diff_out = diff_model(context, batch_paths, return_noised=True)
            diff_metrics = diff_loss_fn(diff_out, actual)
            diff_metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
            diff_opt.step()

        dist_scheduler.step()
        diff_scheduler.step()

        dist_val = validate_model(dist_model, val_loader, dist_loss_fn, device, "DIST", is_diffusion=False)
        diff_val = validate_model(diff_model, val_loader, diff_loss_fn, device, "DIFF", is_diffusion=True)

        def g(h, k): return h.get(k, 0.0)
        dist_corr = g(dist_val, "corr_with_actual")
        diff_corr = g(diff_val, "corr_with_actual")

        logger.info(f"  DIST     val: corr={dist_corr:.4f} dir_acc={g(dist_val,'dir_accuracy'):.4f} loss={g(dist_val,'loss'):.6f}")
        logger.info(f"  DIFFUSION val: corr={diff_corr:.4f} dir_acc={g(diff_val,'dir_accuracy'):.4f} loss={g(diff_val,'loss'):.6f}")

        if dist_corr > best_dist_corr:
            best_dist_corr = dist_corr
            torch.save({"epoch": epoch, "model_state_dict": dist_model.state_dict(), "val": dist_val}, output_dir / "best_distribution_selector.pt")
            logger.info(f"  [SAVED] best dist (corr={best_dist_corr:.4f})")

        if diff_corr > best_diff_corr:
            best_diff_corr = diff_corr
            torch.save({"epoch": epoch, "model_state_dict": diff_model.state_dict(), "val": diff_val}, output_dir / "best_diffusion_selector.pt")
            logger.info(f"  [SAVED] best diffusion (corr={best_diff_corr:.4f})")

        history.append({"epoch": epoch, "dist": dist_val, "diffusion": diff_val})

    torch.save({"epoch": args.epochs, "model_state_dict": diff_model.state_dict()}, output_dir / "final_diffusion_selector.pt")

    final_dist = history[-1]["dist"]
    final_diff = history[-1]["diffusion"]

    def gc(h, k): return max(g(history[i][h], "corr_with_actual") for i in range(len(history)))
    best_d = gc("dist", "corr_with_actual")
    best_df = gc("diffusion", "corr_with_actual")

    corr_imp = (best_df - best_d) / (best_d + 1e-6)
    final_dir_d = g(final_dist, "dir_accuracy")
    final_dir_df = g(final_diff, "dir_accuracy")
    dir_imp = final_dir_df - final_dir_d

    adopt = best_df >= 0.30 and final_dir_df >= 0.61

    summary = {
        "best_dist_corr": best_d,
        "best_diff_corr": best_df,
        "dist_dir_acc": final_dir_d,
        "diff_dir_acc": final_dir_df,
        "corr_improvement_pct": corr_imp * 100,
        "dir_improvement": dir_imp,
        "REPLACE_CONDITION": {
            "corr_ge_30": best_df >= 0.30,
            "dir_acc_ge_61": final_dir_df >= 0.61,
        },
        "VERDICT": "ADOPT DIFFUSION_SELECTOR" if adopt else "KEEP DISTRIBUTION_SELECTOR",
    }

    with open(output_dir / "diffusion_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n========== COMPARISON SUMMARY ==========")
    logger.info(f"  DIST      best_corr={best_d:.4f}  final_dir={final_dir_d:.4f}")
    logger.info(f"  DIFFUSION best_corr={best_df:.4f}  final_dir={final_dir_df:.4f}")
    logger.info(f"  Corr improvement: {corr_imp*100:+.2f}%")
    logger.info(f"  Dir improvement:  {dir_imp:+.4f}")
    logger.info(f"  VERDICT: {summary['VERDICT']}")
    logger.info(f"=========================================")


if __name__ == "__main__":
    main()