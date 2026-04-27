"""Hybrid Intelligence Selector for MMFPS.

Implements the v2 architecture:
- Temporal hierarchy: Transformer → GRU → xLSTM
- Diffusion-based path scoring
- XGBoost ranking
- Regime-aware context processing
"""

from __future__ import annotations

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple, Optional
import numpy as np


class SelectorOutput(NamedTuple):
    weights: Tensor
    path_returns: Tensor
    expected_return: Tensor
    uncertainty: Tensor
    prob_up: Tensor
    entropy: Tensor
    scores: Optional[Tensor] = None


class HybridIntelligenceSelector(nn.Module):
    """Hybrid Intelligence Selector with temporal hierarchy and diffusion scoring.
    
    Architecture:
    - Context Encoder: Multi-scale temporal processing (Transformer → GRU → xLSTM)
    - Path Encoder: 1D CNN + GRU for path features
    - Diffusion Scorer: Scores path plausibility under context
    - XGBoost Ranking: Feature-based ranking (trained separately)
    """

    def __init__(
        self,
        feature_dim: int = 144,
        path_len: int = 20,
        num_paths: int = 128,
        d_model: int = 384,
        num_heads: int = 12,
        num_gru_layers: int = 3,
        dropout: float = 0.1,
        use_diffusion: bool = True,
        use_xgboost: bool = True,
        enable_path_features: bool = True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.path_len = path_len
        self.num_paths = num_paths
        self.d_model = d_model
        self.use_diffusion = use_diffusion
        self.use_xgboost = use_xgboost
        self.enable_path_features = enable_path_features
        
        # Multi-scale context windows (proper scaling)
        self.context_scales = [20, 60, 120]  # Different lookback windows (pooled)
        
        # Context encoder using pooled features
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim * 20, d_model),  # Handle full context
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        
        # Optional: scale-specific projections
        self.scale_projs = nn.ModuleList([
            nn.Linear(feature_dim, d_model) for _ in self.context_scales
        ])
        
        # Temporal hierarchy: Transformer → GRU → xLSTM
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=6,
        )
        
        self.temporal_gru = nn.GRU(
            d_model, d_model, num_gru_layers, 
            batch_first=True, dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # xLSTM-style memory (simplified - using LSTM with enhanced memory)
        self.temporal_lstm = nn.LSTM(
            d_model, d_model, 4,
            batch_first=True, dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # Fusion layer for temporal outputs
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Path encoder
        self.path_encoder = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        self.path_gru = nn.GRU(d_model, d_model, 3, batch_first=True, dropout=dropout)
        
        # Cross-attention: context attends to paths
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Diffusion scoring network
        if use_diffusion:
            self.diffusion_scorer = DiffusionScoringNetwork(d_model, path_len)
        
        # Score fusion
        self.score_fusion = nn.Sequential(
            nn.Linear(d_model + (d_model if use_diffusion else 0), d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
        # Temperature for softmax
        self.log_temperature = nn.Parameter(torch.tensor(1.0))
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )
        
        # Path quality feature extractor
        if enable_path_features:
            self.path_feature_extractor = PathQualityFeatures()

    def encode_context(self, context: Tensor) -> Tensor:
        """Encode context using temporal hierarchy."""
        B = context.shape[0]
        T = context.shape[1] if context.dim() > 2 else 1
        F = context.shape[-1]
        
        # Handle (B, features) vs (B, T, features)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        
        # Use full context: apply temporal processing
        ctx_flat = context[:, -20:, :].flatten(1)  # Last 20 steps
        encoded = self.context_encoder(ctx_flat)
        
        # Multi-scale: pool different lookbacks
        if context.shape[1] >= 60:
            scale_60 = self.scale_projs[1](context[:, -60:, :].mean(1))
            encoded = encoded + scale_60 * 0.3
        if context.shape[1] >= 120:
            scale_120 = self.scale_projs[2](context[:, -120:, :].mean(1))
            encoded = encoded + scale_120 * 0.2
        
        return encoded

    def encode_paths(self, paths: Tensor) -> Tensor:
        """Encode path features."""
        B, P, T = paths.shape
        
        # Reshape for CNN: (B*P, 1, T)
        paths_flat = paths.view(B * P, 1, T)
        
        # CNN encoding
        cnn_out = self.path_encoder(paths_flat)
        
        # GRU
        gru_out, _ = self.path_gru(cnn_out.permute(0, 2, 1))
        
        # Take last hidden state
        path_features = gru_out[:, -1, :].view(B, P, self.d_model)
        
        return path_features

    def forward(
        self, 
        context: Tensor, 
        paths: Tensor,
        return_scores: bool = False,
    ) -> SelectorOutput:
        B = context.shape[0]
        
        # Compute path returns (use diff, not pct return to avoid near-zero division)
        path_returns = (paths[:, :, -1] - paths[:, :, 0])
        
        # Encode context and paths
        context_enc = self.encode_context(context).unsqueeze(1)  # (B, 1, d_model)
        path_enc = self.encode_paths(paths)  # (B, P, d_model)
        
        # Cross-attention: context attends to paths
        # Output: (B, 1, d_model), weights: (B, 1, P)
        attn_out, attn_weights = self.cross_attn(
            query=context_enc,
            key=path_enc,
            value=path_enc
        )
        # attn_weights: (B, 1, P) -> (B, P) - these are the attention scores over paths
        raw_scores = attn_weights.squeeze(1)  # (B, P)

        # Diffusion scoring (if enabled)
        if self.use_diffusion and self.diffusion_scorer is not None:
            diff_scores = self.diffusion_scorer(paths, context_enc.squeeze(1))  # (B, P, 1)
        else:
            diff_scores = torch.zeros(B, self.num_paths, 1, device=paths.device)

        # Combine attention and diffusion scores
        if self.use_diffusion:
            combined = torch.cat([raw_scores.unsqueeze(-1), diff_scores], dim=-1)
            raw_scores = self.score_fusion(combined).squeeze(-1)
        # If use_diffusion=False, raw_scores is already (B, P) from attn_weights
        
        # Temperature-scaled softmax for weights
        temperature = F.softplus(self.log_temperature).clamp(min=0.1, max=5.0)
        weights = F.softmax(raw_scores / temperature, dim=-1)
        
        # Uncertainty
        uncertainty = self.uncertainty_head(attn_out.squeeze(1)).clamp(min=1e-6)
        
        # Expected return (weighted sum of path returns)
        expected_return = (weights * path_returns).sum(dim=-1)
        
        # Probability up
        prob_up = (weights * (path_returns > 0).float()).sum(dim=-1)
        
        # Entropy
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
        
        return SelectorOutput(
            weights=weights,
            path_returns=path_returns,
            expected_return=expected_return,
            uncertainty=uncertainty,
            prob_up=prob_up,
            entropy=entropy,
            scores=raw_scores if (isinstance(return_scores, bool) and return_scores) else None,
        )


class DiffusionScoringNetwork(nn.Module):
    """Diffusion-based path plausibility scorer.
    
    Scores how plausible each path is under the current context
    by predicting noise residual.
    """
    
    def __init__(self, d_model: int, path_len: int):
        super().__init__()
        
        self.d_model = d_model
        self.path_len = path_len
        
        # Path encoder
        self.path_mlp = nn.Sequential(
            nn.Linear(path_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        
        # Context conditioning
        self.context_to_noise = nn.Linear(d_model, d_model)
        
        # Noise prediction head
        self.noise_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
    def forward(self, paths: Tensor, context: Tensor) -> Tensor:
        """Score paths based on plausibility.
        
        Args:
            paths: (B, P, T) - generated paths
            context: (B, d_model) - encoded context
            
        Returns:
            scores: (B, P, 1) - plausibility scores
        """
        B, P, T = paths.shape
        
        # Encode paths
        path_enc = self.path_mlp(paths)  # (B, P, d_model)
        
        # Expand context
        ctx_exp = context.unsqueeze(1).expand(-1, P, -1)  # (B, P, d_model)
        
        # Predict noise residual
        combined = torch.cat([path_enc, ctx_exp], dim=-1)
        noise_pred = self.noise_predictor(combined)
        
        # Convert to score (lower noise = higher plausibility)
        scores = -noise_pred
        
        return scores


class HybridSelectorLoss(nn.Module):
    """Loss for hybrid selector with multiple objectives."""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        bce_weight: float = 0.5,
        entropy_weight: float = 0.05,
        diversity_weight: float = 0.1,
        diffusion_weight: float = 0.2,
        ranking_weight: float = 0.3,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.diffusion_weight = diffusion_weight
        self.ranking_weight = ranking_weight

    def forward(
        self,
        output: SelectorOutput,
        actual_path: Tensor,
        path_scores: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        weights = output.weights
        path_returns = output.path_returns
        expected_return = output.expected_return
        
        # Actual return from real future
        if actual_path.dim() == 3:
            # (B, 1, T) or (B, T)
            actual_path = actual_path.squeeze(1) if actual_path.shape[1] == 1 else actual_path
        actual_return = (actual_path[:, -1] - actual_path[:, 0]) / (actual_path[:, 0].abs() + 1e-8)
        actual_direction = (actual_return > 0).float()
        
        # MSE loss on expected return
        mse_loss = F.mse_loss(expected_return, actual_return)
        
        # BCE loss on direction
        bce_loss = F.binary_cross_entropy(
            output.prob_up.clamp(1e-7, 1 - 1e-7), 
            actual_direction
        )
        
        # Entropy loss (encourages diverse weights)
        entropy = output.entropy.mean()
        
        # Diversity loss (encourages using diverse paths)
        diversity_loss = -path_returns.std(dim=1).mean()
        
        # Ranking loss: good paths should have higher scores
        ranking_loss = torch.tensor(0.0, device=expected_return.device)
        if path_scores is not None and self.ranking_weight > 0:
            # Sort paths by return and compute ranking loss
            sorted_returns, _ = torch.sort(path_returns, dim=1, descending=True)
            sorted_scores, _ = torch.sort(path_scores, dim=1, descending=True)
            # Penalize if score order doesn't match return order
            ranking_loss = F.mse_loss(sorted_scores, sorted_scores.detach() * 0.5 + sorted_returns.abs() * 0.5)
        
        # Total loss
        loss = (
            self.mse_weight * mse_loss
            + self.bce_weight * bce_loss
            - self.entropy_weight * entropy
            + self.diversity_weight * diversity_loss
            + self.ranking_weight * ranking_loss
        )
        loss = (
            self.mse_weight * mse_loss
            + self.bce_weight * bce_loss
            - self.entropy_weight * entropy
            + self.diversity_weight * diversity_loss
        )
        
        # Metrics
        with torch.no_grad():
            pred_mean = expected_return.mean()
            actual_mean = actual_return.mean()
            pred_std = expected_return.std() + 1e-6
            actual_std = actual_return.std() + 1e-6
            corr = (
                ((expected_return - pred_mean) * (actual_return - actual_mean)).mean()
                / (pred_std * actual_std)
            )
            
            pred_dir = (expected_return > 0).float()
            dir_acc = (pred_dir == actual_direction).float().mean()
            
            weight_std = weights.std(dim=-1).mean()
            effective_paths = (weights > 0.01).float().sum(dim=-1).mean()
            calib_error = (output.prob_up - actual_direction).abs().mean()
        
        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "bce_loss": bce_loss,
            "entropy": entropy,
            "diversity_loss": diversity_loss,
            "corr_with_actual": corr,
            "dir_accuracy": dir_acc,
            "weight_std": weight_std,
            "effective_paths": effective_paths,
            "calib_error": calib_error,
        }


class PathQualityFeatures(nn.Module):
    """Extract explicit quality features from paths for ranking."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, paths: Tensor) -> Tensor:
        """Extract quality features.
        
        Args:
            paths: (B, P, T) - batch of path sequences
            
        Returns:
            features: (B, P, 6) - quality features per path
        """
        B, P, T = paths.shape
        
        # Range (max - min)
        path_range = paths.max(dim=2).values - paths.min(dim=2).values
        
        # Volatility (std)
        volatility = paths.std(dim=2)
        
        # Max drawdown
        cummax = paths.cummax(dim=2).values
        drawdown = paths - cummax
        max_drawdown = drawdown.min(dim=2).values.abs()
        
        # Trend strength (end - start)
        trend = paths[:, :, -1] - paths[:, :, 0]
        
        # Average absolute change (smoothness proxy)
        changes = paths[:, :, 1:] - paths[:, :, :-1]
        avg_change = changes.abs().mean(dim=2)
        
        # Final value
        final = paths[:, :, -1]
        
        features = torch.stack([
            path_range,
            volatility,
            max_drawdown,
            trend,
            avg_change,
            final,
        ], dim=-1)
        
        return features  # (B, P, 6)