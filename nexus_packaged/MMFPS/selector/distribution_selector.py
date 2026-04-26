"""Distribution-aware path selector for MMFPS.

The ONLY pathway to predict return is via weighted path return aggregation.
No context-only shortcut exists.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple


class SelectorOutput(NamedTuple):
    weights: Tensor
    path_returns: Tensor
    expected_return: Tensor
    uncertainty: Tensor
    prob_up: Tensor
    entropy: Tensor


class DistributionSelector(nn.Module):
    """Distribution-aware selector that MUST use path returns.

    Architecture constraint: return = weighted_sum(path_returns) ONLY.
    Context feeds attention, attention weights path returns, nothing else.
    """

    def __init__(
        self,
        feature_dim: int = 144,
        path_dim: int = 20,
        num_paths: int = 64,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_paths = num_paths
        self.path_dim = path_dim
        self.d_model = d_model

        self.context_enc = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.path_enc = nn.Sequential(
            nn.Linear(path_dim, d_model),
            nn.GELU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.attn_residual = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.log_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, context: Tensor, paths: Tensor) -> SelectorOutput:
        B = context.shape[0]

        path_returns = (paths[:, :, -1] - paths[:, :, 0]) / (paths[:, :, 0] + 1e-8)

        C = self.context_enc(context).unsqueeze(1)
        P = self.path_enc(paths)

        attn_out, attn_weights = self.cross_attn(query=C, key=P, value=P)
        attn_weights = attn_weights.squeeze(1)

        temperature = F.softplus(self.log_temperature).clamp(min=0.1, max=5.0)
        weights = F.softmax(attn_weights / temperature, dim=-1)

        attn_residual = self.attn_residual(C).squeeze(1)
        attn_features = attn_out.squeeze(1)

        uncertainty_raw = self.uncertainty_head(attn_features + attn_residual)
        uncertainty = F.softplus(uncertainty_raw).clamp(min=1e-6, max=1.0)

        expected_return = (weights * path_returns).sum(dim=-1)

        prob_up = (weights * (path_returns > 0).float()).sum(dim=-1)

        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)

        return SelectorOutput(
            weights=weights,
            path_returns=path_returns,
            expected_return=expected_return,
            uncertainty=uncertainty,
            prob_up=prob_up,
            entropy=entropy,
        )


class SelectorLoss(nn.Module):
    """Loss that forces path usage via weighted return aggregation.

    NO path labels. NO cross-entropy. NO argmax.
    Only MSE on weighted sum and BCE on direction.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        bce_weight: float = 0.5,
        entropy_weight: float = 0.05,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight

    def forward(
        self,
        output: SelectorOutput,
        actual_path: Tensor,
    ) -> dict[str, Tensor]:
        weights = output.weights
        path_returns = output.path_returns
        expected_return = output.expected_return

        actual_return = (actual_path[:, -1] - actual_path[:, 0]) / (actual_path[:, 0] + 1e-8)
        actual_direction = (actual_return > 0).float()

        mse_loss = F.mse_loss(expected_return, actual_return)

        bce_loss = F.binary_cross_entropy(
            output.prob_up.clamp(1e-7, 1 - 1e-7), actual_direction
        )

        entropy = output.entropy.mean()

        diversity_loss = -path_returns.std(dim=1).mean()

        loss = (
            self.mse_weight * mse_loss
            + self.bce_weight * bce_loss
            - self.entropy_weight * entropy
            + self.diversity_weight * diversity_loss
        )

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