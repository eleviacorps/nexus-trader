"""V30 Inference - run trained evaluator on live data."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from nexus_packaged.v30.models.evaluator.evaluator import PathEvaluator, path_to_scores
from nexus_packaged.v30.inference.aggregator import Aggregator, TradingSignal


class V30InferenceEngine:
    """V30 inference engine using trained evaluator."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        device: str = "cuda",
    ):
        """
        Args:
            checkpoint_path: Path to trained evaluator checkpoint
            config: Model configuration
            device: cuda or cpu
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("v30.inference")
        
        # Load evaluator
        self.evaluator = self._load_evaluator(checkpoint_path)
        
        # Setup aggregator
        eval_config = config.get("evaluation", {})
        self.aggregator = Aggregator(
            confidence_threshold=eval_config.get("confidence_threshold", 0.55),
            min_ev_threshold=eval_config.get("min_ev_threshold", 0.0001),
        )
        
        # Model config
        self.num_paths = config.get("model", {}).get("generator", {}).get("num_paths", 64)
        self.horizon = config.get("model", {}).get("generator", {}).get("horizon", 20)
        self.lookback = config.get("model", {}).get("generator", {}).get("lookback", 128)
        
    def _load_evaluator(self, checkpoint_path: str) -> PathEvaluator:
        """Load trained evaluator from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        eval_cfg = self.config.get("model", {}).get("evaluator", {})
        
        evaluator = PathEvaluator(
            feature_dim=eval_cfg.get("hidden_dim", 256),
            path_dim=self.config.get("model", {}).get("generator", {}).get("horizon", 20),
            num_paths=self.config.get("model", {}).get("generator", {}).get("num_paths", 64),
            hidden_dim=eval_cfg.get("hidden_dim", 256),
            num_layers=eval_cfg.get("num_layers", 3),
            dropout=eval_cfg.get("dropout", 0.1),
        )
        
        evaluator.load_state_dict(checkpoint["model_state_dict"])
        evaluator.to(self.device)
        evaluator.eval()
        
        self.logger.info(f"Loaded evaluator from {checkpoint_path}")
        
        return evaluator
    
    def predict(
        self,
        context: np.ndarray,
        paths: np.ndarray,
    ) -> TradingSignal:
        """
        Generate trading signal from context and paths.
        
        Args:
            context: [lookback, feature_dim] - market context
            paths: [num_paths, horizon] - generated diffusion paths
        
        Returns:
            TradingSignal
        """
        # Prepare inputs
        context_t = torch.from_numpy(context).float().unsqueeze(0).to(self.device)
        paths_t = torch.from_numpy(paths).float().unsqueeze(0).to(self.device)
        
        # Get evaluator weights
        with torch.no_grad():
            scores = self.evaluator(context_t, paths_t)
            weights = torch.softmax(scores, dim=-1).cpu().numpy()[0]
        
        # Aggregate to trading signal
        signal = self.aggregator.aggregate(
            paths=paths,
            weights=weights,
        )
        
        return signal
    
    def predict_batch(
        self,
        contexts: np.ndarray,
        paths_batch: np.ndarray,
    ) -> list[TradingSignal]:
        """Process batch of contexts and paths."""
        B = contexts.shape[0]
        
        context_t = torch.from_numpy(contexts).float().to(self.device)
        paths_t = torch.from_numpy(paths_batch).float().to(self.device)
        
        with torch.no_grad():
            scores = self.evaluator(context_t, paths_t)
            weights = torch.softmax(scores, dim=-1).cpu().numpy()
        
        signals = []
        for paths, w in zip(paths_batch, weights):
            signal = self.aggregator.aggregate(paths=paths, weights=w)
            signals.append(signal)
        
        return signals


def run_live_inference(
    checkpoint_path: str,
    config_path: str,
    output_path: str | None = None,
):
    """Run live inference with V30 engine."""
    import yaml
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    logger = logging.getLogger("v30.inference")
    
    # Create engine
    device = config.get("inference", {}).get("device", "cuda")
    engine = V30InferenceEngine(checkpoint_path, config, device)
    
    logger.info("V30 inference engine ready")
    
    # Example: would connect to live data feed here
    # For now, just report ready
    
    return engine


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="V30 Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="nexus_packaged/v30/configs/v30_config.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    engine = run_live_inference(args.checkpoint, args.config, args.output)
    
    print("V30 Inference Engine Ready")
    print(f"Model: {args.checkpoint}")


if __name__ == "__main__":
    main()