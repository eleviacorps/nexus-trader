"""Precompute diffusion paths for V30 training - GPU accelerated.

This creates a cached dataset of paths so training doesn't need to 
run the diffusion model repeatedly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent to path for imports
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import torch

from nexus_packaged.core.diffusion_loader import DiffusionModelLoader
from nexus_packaged.protection.encryptor import derive_key_from_env


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("v30.precompute")


def load_diffusion_model(config: dict, device: str = "cuda"):
    """Load the frozen diffusion model."""
    logger = logging.getLogger("v30.precompute")
    
    model_path = config.get("model", {}).get("generator", {}).get("path",
        "models/freezed_diff/v26/diffusion_unet1d_v2_regime.pt")
    
    # Use encryption key
    key = derive_key_from_env(
        env_var="NEXUS_MODEL_KEY",
        salt="nexus_trader_salt_2024",
    )
    
    # Override device in config
    config = config.copy()
    config["model"] = config.get("model", {}).copy()
    config["model"]["generator"] = config.get("model", {}).get("generator", {}).copy()
    config["model"]["generator"]["device"] = device
    
    loader = DiffusionModelLoader(model_path, key, settings=config)
    loader.load()
    loader.warm_up()
    
    logger.info(f"Loaded diffusion model from {model_path} on {device}")
    return loader


def precompute_paths(
    features: np.ndarray,
    model_loader: DiffusionModelLoader,
    config: dict,
    output_path: str,
    lookback: int = 128,
    num_paths: int = 64,
    horizon: int = 20,
    start_idx: int | None = None,
    end_idx: int | None = None,
    batch_size: int = 64,
    chunk_size: int = 100000,
    minute_interval: int = 15,
):
    """Precompute all paths and save to disk using GPU batching with chunked processing.
    
    Args:
        features: Feature array [N, feature_dim] - 1-minute point-in-time features
        model_loader: Loaded diffusion model
        config: Configuration dict
        output_path: Where to save paths
        lookback: Context length in 15-minute candles
        num_paths: Number of paths per sample
        horizon: Horizon length in 15-minute candles  
        start_idx, end_idx: Range to process (in 15-minute samples)
        batch_size: GPU batch size for efficiency
        chunk_size: Number of samples to process before saving
        minute_interval: Data interval (15 for 15-min candles)
    """
    logger = logging.getLogger("v30.precompute")
    
    # Resample features to 15-minute intervals (take every 15th)
    step = minute_interval
    features_15min = features[::step]  # Every 15th row
    logger.info(f"Resampled features from {features.shape[0]} to {features_15min.shape[0]} (15-min intervals)")
    
    # Now lookback and horizon are in 15-minute units directly
    # Total context: lookback * 15 minutes of history
    # Prediction: horizon * 15 minutes ahead
    
    if start_idx is None:
        start_15min = lookback
    else:
        start_15min = max(start_idx, lookback)
    
    if end_idx is None:
        end_15min = len(features_15min) - horizon - 1
    else:
        end_15min = min(end_idx, len(features_15min) - horizon - 1)
    
    num_samples = max(0, end_15min - start_15min)
    
    logger.info(f"Precomputing {num_samples} samples (15-minute interval)...")
    logger.info(f"Features: {features_15min.shape}, lookback={lookback} ({lookback*step} min), num_paths={num_paths}, horizon={horizon} ({horizon*step} min)")
    logger.info(f"Batch size: {batch_size}, Chunk size: {chunk_size}")
    
    # Pre-create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-allocate array
    paths = np.zeros((num_samples, num_paths, horizon), dtype=np.float32)
    
    start_time = time.time()
    samples_done = 0
    
    for chunk_start in range(0, num_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk_samples = chunk_end - chunk_start
        
        logger.info(f"Processing chunk {chunk_start}-{chunk_end} ({chunk_samples} samples)...")
        
        # Process this chunk in batches
        for i in range(chunk_start, chunk_end, batch_size):
            batch_end = min(i + batch_size, chunk_end)
            batch_count = batch_end - i
            
            # Build batch of context windows
            # Each sample needs [lookback, feature_dim]
            batch_contexts = []
            for j in range(i, batch_end):
                # Context for sample j is features from (start_15min + j - lookback) to (start_15min + j)
                ctx = features_15min[start_15min + j - lookback : start_15min + j]
                batch_contexts.append(ctx)
            batch_contexts = np.stack(batch_contexts, axis=0)  # [B, lookback, feature_dim]
            
            # Generate paths for batch
            batch_paths = model_loader.predict(batch_contexts)
            
            # Store results
            paths[i:batch_end] = batch_paths
            
            samples_done += batch_count
            
            # Progress every 10k samples
            if (i + batch_size) % 10000 < batch_size:
                elapsed = time.time() - start_time
                progress = samples_done / num_samples
                eta = elapsed / progress * (1 - progress) if progress > 0 else 0
                logger.info(f"Progress: {samples_done}/{num_samples} ({100*progress:.1f}%) | "
                           f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    # Save
    np.save(output_path, paths)
    
    total_time = time.time() - start_time
    logger.info(f"Saved paths to {output_path}")
    logger.info(f"Final shape: {paths.shape}")
    logger.info(f"Total time: {total_time:.1f}s ({num_samples/total_time:.1f} samples/sec)")


def main():
    parser = argparse.ArgumentParser(description="Precompute V30 training paths (GPU)")
    parser.add_argument("--config", type=str, default="nexus_packaged/v30/configs/v30_config.yaml")
    parser.add_argument("--output", type=str, default="nexus_packaged/v30/data/processed/v30_paths.npy")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--num-paths", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256, help="GPU batch size")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Samples per chunk (memory management)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    logger = setup_logging()
    
    # Load features
    data_config = config.get("data", {})
    features_path = data_config.get("features_path", "data/features/diffusion_fused_6m.npy")
    
    # Try multiple possible paths
    candidates = [
        Path(features_path),
        Path("data/features/diffusion_fused_6m.npy"),
        Path("nexus_packaged/data/features/diffusion_fused_6m.npy"),
        Path("data/features/diffusion_fused_405k.npy"),
    ]
    
    features = None
    for p in candidates:
        if p.exists():
            features = np.load(p)
            logger.info(f"Loaded features from {p}: {features.shape}")
            break
    
    if features is None:
        raise FileNotFoundError(f"Could not find features file. Tried: {[str(c) for c in candidates]}")
    
    # Validate feature dim
    model_cfg = config.get("model", {}).get("generator", {})
    expected_feature_dim = model_cfg.get("feature_dim", 144)
    lookback = model_cfg.get("lookback", 128)
    horizon = model_cfg.get("horizon", 20)
    
    # Features are [N, feature_dim] - need to create sliding windows
    # For each sample i, context = features[i-lookback:i]
    if features.ndim == 2 and features.shape[1] == expected_feature_dim:
        # Already in correct format: [N, feature_dim]
        # Will create context windows on-the-fly during generation
        logger.info(f"Features shape: {features.shape} (N, feature_dim)")
        total_samples = len(features) - lookback - horizon
        logger.info(f"Total valid samples: {total_samples} (from index {lookback} to {len(features)-horizon-1})")
    
    # Limit samples if requested (in 15-min samples)
    end_idx_15min = args.end
    if args.limit:
        # Resample limit to 15-min units
        limit_15min = args.limit // 15
        end_idx_15min = min(limit_15min, (len(features) // 15) - horizon - 1)
    
    # Load model on GPU
    logger.info(f"Loading diffusion model on {args.device}...")
    model_loader = load_diffusion_model(config, device=args.device)
    
    # Precompute
    horizon = args.horizon
    num_paths = args.num_paths
    
    precompute_paths(
        features=features,
        model_loader=model_loader,
        config=config,
        output_path=args.output,
        lookback=lookback,
        num_paths=num_paths,
        horizon=horizon,
        start_idx=args.start,
        end_idx=end_idx_15min,
        batch_size=args.batch_size,
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()