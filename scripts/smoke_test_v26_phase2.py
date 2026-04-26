"""Minimal smoke test for V26 Phase 2 - skips slow diffusion, tests only the training loop."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    V24_DIFFUSION_FUSED_6M_PATH,
    V24_DIFFUSION_NORM_STATS_6M_PATH,
    V24_DIFFUSION_TIMESTAMPS_6M_PATH,
    OUTPUTS_V26_DIR,
)
from src.v24.diffusion.dataset import split_by_year
from src.v26.diffusion.regime_dataset import RegimeDiffusionDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_dummy_epoch(
    dummy_state,
    dataloader,
    optimizer,
    device,
    epoch,
):
    """Dummy training that just uses random tensors instead of slow diffusion."""
    total_loss = 0.0
    total_consistency = 0.0
    n_batches = 0

    for batch_idx, (window, past_ctx, regime_probs) in enumerate(dataloader):
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)
        regime_probs = regime_probs.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Fake loss that depends on dummy parameters for grad
        output = dummy_state.short_weight.mean() * past_ctx.shape[0]
        output = output + dummy_state.medium_weight.mean() * 0
        loss = torch.abs(output)  # non-zero loss that requires grad

        loss.backward()
        torch.nn.utils.clip_grad_norm_(dummy_state.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_consistency += 0.5  # dummy value
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}, consistency={avg_consistency.item():.4f}")

    return {
        "total_loss": total_loss / max(n_batches, 1),
        "consistency": total_consistency / max(n_batches, 1),
    }


@torch.no_grad()
def _validate_dummy(
    dataloader,
    device,
) -> dict:
    """Dummy validation."""
    total_consistency = 0.0
    n_batches = 0

    for window, past_ctx, regime_probs in dataloader:
        window = window.to(device, non_blocking=True)
        past_ctx = past_ctx.to(device, non_blocking=True)
        regime_probs = regime_probs.to(device, non_blocking=True)

        # Fake consistency based on data
        avg_consistency = torch.sigmoid(past_ctx.mean() * 0.001)
        total_consistency += avg_consistency.item()
        n_batches += 1

    return {
        "val_consistency": total_consistency / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="V26 Phase 2 Smoke Test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    _set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    print("=" * 60)
    print("V26 Phase 2: Smoke Test (skip diffusion)")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples: {args.max_samples}")
    print(f"Device: {device}")
    print()

    # Dummy model state
    dummy_state = torch.nn.Module()
    dummy_state.short_weight = torch.nn.Parameter(torch.randn(64, 64, device=device) * 0.01)
    dummy_state.medium_weight = torch.nn.Parameter(torch.randn(64, 64, device=device) * 0.01)

    optimizer = AdamW(dummy_state.parameters(), lr=args.lr, weight_decay=0.01)

    # Load dataset with caching
    print("Loading dataset...")
    fused_path = V24_DIFFUSION_FUSED_6M_PATH
    timestamps_path = V24_DIFFUSION_TIMESTAMPS_6M_PATH
    timestamps = None
    if timestamps_path.exists():
        timestamps = np.load(str(timestamps_path), mmap_mode="r")

    total_rows = 6024602
    train_slice, val_slice, _ = split_by_year(total_rows, 120, timestamps=timestamps)
    print(f"Train: {len(train_slice):,}, Val: {len(val_slice):,}")

    regime_cache_path = PROJECT_ROOT / "data" / "cache" / "v26_regime_labels_tactical_v24_2.npy"

    train_ds = RegimeDiffusionDataset(
        fused_path, 120, train_slice,
        context_len=256, max_samples=args.max_samples, load_to_ram=True,
        regime_cache_path=regime_cache_path,
    )
    val_ds = RegimeDiffusionDataset(
        fused_path, 120, val_slice,
        context_len=256, max_samples=min(1000, len(val_slice)), load_to_ram=True,
        regime_cache_path=regime_cache_path,
    )

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    log_path = OUTPUTS_V26_DIR / "smoke_test_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_consistency = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs - 1}")

        train_metrics = _train_dummy_epoch(
            dummy_state, train_dl, optimizer, device, epoch
        )

        val_metrics = _validate_dummy(val_dl, device)

        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["total_loss"],
            "train_consistency": train_metrics["consistency"],
            "val_consistency": val_metrics["val_consistency"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }

        if val_metrics["val_consistency"] > best_consistency:
            best_consistency = val_metrics["val_consistency"]
            entry["best"] = True

        with open(str(log_path), "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  Train: loss={train_metrics['total_loss']:.4f}, consistency={train_metrics['consistency']:.4f}")
        print(f"  Val: consistency={val_metrics['val_consistency']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        if entry.get("best"):
            print(f"  [BEST] consistency={best_consistency:.4f}")

    print(f"\n{'=' * 60}")
    print(f"Smoke test complete!")
    print(f"Best consistency: {best_consistency:.4f}")
    print(f"Logs: {log_path}")


if __name__ == "__main__":
    main()