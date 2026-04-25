"""Batched path generator with caching for V30 training."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch


class CachedPathGenerator:
    """Path generator with caching for efficient training.
    
    This addresses the critical efficiency issue:
    - Pre-compute paths for training data
    - Cache on disk to avoid regenerating every epoch
    - Deterministic (seeded) for reproducibility
    """
    
    def __init__(
        self,
        generator_fn: Callable[[np.ndarray], np.ndarray],
        cache_dir: str = "nexus_packaged/v30/data/processed",
        seed: int = 42,
    ):
        """
        Args:
            generator_fn: Function that takes (lookback, feature_dim) returns (num_paths, horizon)
            cache_dir: Where to store cached paths
            seed: Random seed for reproducibility
        """
        self.generator_fn = generator_fn
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self._logger = logging.getLogger("v30.generator")
        
    def _get_cache_key(self, context: np.ndarray) -> str:
        """Generate cache key from context."""
        # Use hash of context for cache key
        context_hash = hashlib.md5(context.tobytes()).hexdigest()[:16]
        return context_hash
    
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"paths_{cache_key}.npy"
    
    def generate(self, context: np.ndarray) -> np.ndarray:
        """Generate paths for a single context (with caching)."""
        cache_key = self._get_cache_key(context)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            return np.load(cache_path)
        
        # Generate new paths
        paths = self.generator_fn(context)
        
        # Cache for future use
        np.save(cache_path, paths)
        
        return paths
    
    def generate_batch(self, contexts: np.ndarray) -> np.ndarray:
        """Generate paths for batch of contexts.
        
        Args:
            contexts: [B, lookback, feature_dim]
        
        Returns:
            paths: [B, num_paths, horizon]
        """
        B = contexts.shape[0]
        all_paths = []
        
        for i in range(B):
            paths = self.generate(contexts[i])
            all_paths.append(paths)
        
        return np.stack(all_paths, axis=0)
    
    def precompute_and_cache(
        self,
        features: np.ndarray,
        start_idx: int,
        end_idx: int,
        lookback: int,
        force: bool = False,
    ) -> int:
        """Pre-compute and cache paths for a range of indices.
        
        Args:
            features: Full feature array
            start_idx: Start index for caching
            end_idx: End index for caching
            lookback: Lookback window size
            force: Re-generate even if cached
        
        Returns:
            Number of paths cached
        """
        cached = 0
        
        for idx in range(start_idx, end_idx):
            context = features[idx - lookback:idx]
            cache_key = self._get_cache_key(context)
            cache_path = self._get_cache_path(cache_key)
            
            if not force and cache_path.exists():
                continue
            
            paths = self.generator_fn(context)
            np.save(cache_path, paths)
            cached += 1
            
            if cached % 100 == 0:
                self._logger.info(f"Cached {cached} paths...")
        
        self._logger.info(f"Pre-computed {cached} new paths")
        return cached
    
    def clear_cache(self) -> int:
        """Clear all cached paths."""
        count = 0
        for f in self.cache_dir.glob("paths_*.npy"):
            f.unlink()
            count += 1
        self._logger.info(f"Cleared {count} cached path files")
        return count


class DeterministicPathGenerator:
    """Wrapper that adds determinism to any path generator.
    
    Ensures same context always produces same paths.
    """
    
    def __init__(self, generator_fn: Callable, base_seed: int = 42):
        self.generator_fn = generator_fn
        self.base_seed = base_seed
        
    def __call__(self, context: np.ndarray) -> np.ndarray:
        """Generate deterministic paths."""
        # Create seed from context hash
        context_hash = hashlib.md5(context.tobytes()).hexdigest()
        seed = int(context_hash[:8], 16) % (2**32)
        
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate
        return self.generator_fn(context)


def create_efficient_generator(
    model_loader,
    config: dict,
    cache_dir: str = "nexus_packaged/v30/data/processed",
    use_cache: bool = True,
    use_deterministic: bool = True,
) -> CachedPathGenerator:
    """Create efficient path generator for training.
    
    Args:
        model_loader: Loaded diffusion model
        config: Model configuration
        cache_dir: Cache directory
        use_cache: Whether to use caching
        use_deterministic: Whether to ensure determinism
    
    Returns:
        CachedPathGenerator ready for training
    """
    def generate_fn(context: np.ndarray) -> np.ndarray:
        """Wrapper around model predict."""
        return model_loader.predict(context)
    
    if use_deterministic:
        generate_fn = DeterministicPathGenerator(generate_fn)
    
    generator = CachedPathGenerator(
        generator_fn=generate_fn,
        cache_dir=cache_dir,
        seed=42,
    )
    
    return generator