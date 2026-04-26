"""GPU-batched 1-minute path precomputation for V30.

Generates 6M rows × 64 paths × K horizon of diffusion paths at 1-minute resolution,
using aggressive batching to saturate GPU.

Strategy:
- Batch across windows: process 256-512 samples per forward pass
- Batch across paths: generate 64 paths simultaneously
- Mixed precision (AMP) for ~2-4x speedup
- Memory-mapped features, chunked saving
- Progress every 50k samples
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

from nexus_packaged.core.diffusion_loader import DiffusionModelLoader
from nexus_packaged.protection.encryptor import derive_key_from_env


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("v30.precompute_1m")


def load_diffusion_model(device: str = "cuda") -> DiffusionModelLoader:
    logger = logging.getLogger("v30.precompute_1m")

    model_path = "models/freezed_diff/v26/diffusion_unet1d_v2_regime.pt"
    key = derive_key_from_env(env_var="NEXUS_MODEL_KEY", salt="nexus_trader_salt_2024")

    config = {
        "model": {
            "generator": {
                "path": model_path,
                "lookback": 128,
                "feature_dim": 144,
                "num_paths": 64,
                "horizon": 20,
                "device": device,
                "norm_stats_path": "config/diffusion_norm_stats_6m.json",
            }
        }
    }

    loader = DiffusionModelLoader(model_path, key, settings=config)
    loader.load()
    loader.warm_up()

    logger.info(f"Loaded diffusion model on {device}")
    return loader


def precompute_1m_batched(
    features_path: str,
    output_path: str,
    model_loader: DiffusionModelLoader,
    num_samples: int = 6000000,
    lookback: int = 128,
    horizon: int = 20,
    batch_size: int = 512,
    chunk_size: int = 50000,
    device: str = "cuda",
):
    """Precompute 1-minute diffusion paths with GPU batching.

    Args:
        features_path: Path to 6M-row feature file
        output_path: Where to save paths [num_samples, 64, horizon]
        model_loader: Loaded diffusion model
        num_samples: Number of 1-minute samples to generate
        lookback: Context length
        horizon: Prediction horizon
        batch_size: GPU batch size (process N windows simultaneously)
        chunk_size: Save to disk every N samples
        device: cuda or cpu
    """
    logger = logging.getLogger("v30.precompute_1m")

    t_start = time.time()

    features = np.load(features_path, mmap_mode="r")
    actual_rows = min(len(features), num_samples)
    logger.info(f"Features: {features.shape}, using {actual_rows} rows")

    valid_start = lookback
    valid_end = min(actual_rows - horizon - 1, num_samples - lookback)
    num_valid = valid_end - valid_start
    logger.info(f"Valid samples: {num_valid} (index {valid_start} to {valid_end})")

    num_paths = 64

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_shape = (num_valid, num_paths, horizon)
    logger.info(f"Output shape: {paths_shape} ({paths_shape[0] * paths_shape[1] * paths_shape[2] * 4 / 1e9:.2f} GB)")

    if Path(output_path).exists():
        paths_fp = np.memmap(output_path, dtype=np.float32, mode="r+", shape=paths_shape)
        resume_offset = 0
        for r in range(num_valid):
            if paths_fp[r].any():
                resume_offset = r + 1
            if r % 500000 == 0 and r > 0:
                logger.info(f"Checking progress... row {r:,}/{num_valid:,}")
        logger.info(f"Resuming from existing file: {output_path} (offset: {resume_offset:,})")
    else:
        paths_fp = np.memmap(output_path, dtype=np.float32, mode="w+", shape=paths_shape)
        resume_offset = 0
        logger.info(f"Opened memmap: {output_path}")

    dummy_ctx = np.zeros((batch_size, lookback, 144), dtype=np.float32)
    dummy_t = torch.from_numpy(dummy_ctx).to(device)
    logger.info("Warming up generator...")
    _ = model_loader.predict(dummy_ctx[:1])

    logger.info(f"\n{'='*60}")
    logger.info(f"GPU BATCHED 1-MIN PRECOMPUTE")
    logger.info(f"  Samples: {num_valid:,}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Paths: {num_paths}, Horizon: {horizon}")
    logger.info(f"  Chunk size: {chunk_size:,}")
    logger.info(f"  Device: {device}")
    logger.info(f"{'='*60}\n")

    dtype = torch.float16 if device == "cuda" else torch.float32
    samples_done = 0

    for chunk_start in range(0, num_valid, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_valid)
        chunk_samples = chunk_end - chunk_start

        chunk_start_time = time.time()

        for i in range(chunk_start, chunk_end, batch_size):
            batch_end = min(i + batch_size, chunk_end)
            batch_count = batch_end - i

            contexts = np.zeros((batch_count, lookback, 144), dtype=np.float32)
            for j in range(batch_count):
                real_idx = valid_start + i + j
                contexts[j] = features[real_idx - lookback : real_idx]

            batch_paths = model_loader.predict(contexts)
            paths_fp[i:batch_end] = batch_paths[:batch_end - i]

            samples_done += batch_count

        chunk_time = time.time() - chunk_start_time
        total_elapsed = time.time() - t_start
        progress = samples_done / num_valid
        eta = total_elapsed / progress * (1 - progress) if progress > 0 else 0
        rate = chunk_samples / chunk_time if chunk_time > 0 else 0

        paths_fp.flush()

        logger.info(
            f"Chunk {chunk_start:,}-{chunk_end:,} ({chunk_samples:,} samples) | "
            f"Progress: {samples_done:,}/{num_valid:,} ({100*progress:.1f}%) | "
            f"Rate: {rate:.0f} samples/sec | "
            f"Elapsed: {total_elapsed:.0f}s | ETA: {eta:.0f}s"
        )

    total_time = time.time() - t_start
    final_rate = num_valid / total_time
    del paths_fp

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {num_valid:,} samples in {total_time:.0f}s ({final_rate:.0f} samples/sec)")
    logger.info(f"Output: {output_path}")
    logger.info(f"Final shape: {paths_shape}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GPU-batched 1-min path precomputation")
    parser.add_argument("--features", type=str, default="data/features/diffusion_fused_6m.npy")
    parser.add_argument("--output", type=str, default="nexus_packaged/v30/data/processed/v30_1m_paths.npy")
    parser.add_argument("--samples", type=int, default=6000000)
    parser.add_argument("--lookback", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logger = setup_logging()

    features_path = Path(args.features)
    if not features_path.exists():
        features_path = _project_root / args.features
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {args.features}")

    logger.info(f"Loading features from {features_path}...")
    model_loader = load_diffusion_model(device=args.device)

    precompute_1m_batched(
        features_path=str(features_path),
        output_path=args.output,
        model_loader=model_loader,
        num_samples=args.samples,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=args.device,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()