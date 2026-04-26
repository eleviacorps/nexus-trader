"""Evaluation pipeline for MMFPS.

Tracks all key metrics for the simulation-based forecasting system.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

nexus_path = Path(__file__).resolve().parents[2]
if str(nexus_path) not in sys.path:
    sys.path.insert(0, str(nexus_path))
from MMFPS.mmfps import MMFPS


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


class MarketDataset:
    def __init__(self, features, close_prices, start_idx, num_samples):
        self.features = features
        self.close_prices = close_prices
        self.start_idx = start_idx
        self.end_idx = min(start_idx + num_samples, len(features) - 20, len(close_prices) - 20)


def load_data(num_samples=100000):
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    features_path = repo_root / "data/features/diffusion_fused_6m.npy"
    ohlcv_path = repo_root / "nexus_packaged/data/ohlcv.parquet"
    features = np.load(str(features_path), mmap_mode="r")
    ohlcv_df = pd.read_parquet(str(ohlcv_path))
    close_prices = ohlcv_df["close"].values.astype(np.float32)
    return features, close_prices


def evaluate(model, features, close_prices, device, batch_size=512, num_samples=60000):
    model.eval()
    logger = logging.getLogger("mmfps.eval")

    all_preds = []
    all_actuals = []
    all_regime_types = []
    all_weight_entropies = []
    all_path_diversities = []

    for start in tqdm(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)
        contexts = np.array(features[start:end].astype(np.float32))
        context_t = torch.tensor(contexts, device=device)

        with torch.no_grad():
            output = model(context_t)

        actual_starts = np.arange(start, end)
        actual_ends = np.minimum(actual_starts + 21, len(close_prices))
        actuals = []
        for s, e in zip(actual_starts, actual_ends):
            a = close_prices[s:e]
            if len(a) < 21:
                pad = np.zeros(21, dtype=np.float32)
                pad[: len(a)] = a
                a = pad
            actuals.append(a)
        actuals = np.array(actuals)
        actual_returns = (actuals[:, -1] - actuals[:, 0]) / (actuals[:, 0] + 1e-8)

        all_preds.append(output.expected_return.cpu().numpy())
        all_actuals.append(actual_returns)
        all_regime_types.append(output.regime_type.cpu().numpy())
        all_weight_entropies.append(output.weight_entropy.cpu().numpy())
        all_path_diversities.append(output.path_diversity.cpu().numpy())

    preds = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    regime_types = np.concatenate(all_regime_types)
    entropies = np.concatenate(all_weight_entropies)
    diversities = np.concatenate(all_path_diversities)

    pred_mean = preds.mean()
    actual_mean = actuals.mean()
    pred_std = preds.std() + 1e-6
    actual_std = actuals.std() + 1e-6

    corr = float(np.corrcoef(preds, actuals)[0, 1])

    pred_dir = (preds > 0).astype(float)
    actual_dir = (actuals > 0).astype(float)
    dir_acc = float((pred_dir == actual_dir).mean())

    metrics = {
        "num_samples": num_samples,
        "pred_mean": float(pred_mean),
        "actual_mean": float(actual_mean),
        "pred_std": float(pred_std),
        "actual_std": float(actual_std),
        "correlation": corr,
        "directional_accuracy": dir_acc,
        "path_diversity": {
            "mean": float(diversities.mean()),
            "std": float(diversities.std()),
        },
        "weight_entropy": {
            "mean": float(entropies.mean()),
            "std": float(entropies.std()),
        },
        "regime_distribution": {
            "trend_up": float((regime_types == 0).sum() / len(regime_types)),
            "chop": float((regime_types == 1).sum() / len(regime_types)),
            "reversal": float((regime_types == 2).sum() / len(regime_types)),
        },
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path("nexus_packaged/MMFPS/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("mmfps.eval", str(output_dir / "eval.log"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = MMFPS().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    features, close_prices = load_data(args.num_samples)

    metrics = evaluate(
        model, features, close_prices, device,
        batch_size=args.batch_size, num_samples=args.num_samples,
    )

    logger.info(f"\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            logger.info(f"  {k}:")
            for sub_k, sub_v in v.items():
                logger.info(f"    {sub_k}: {sub_v}")
        else:
            logger.info(f"  {k}: {v}")

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()