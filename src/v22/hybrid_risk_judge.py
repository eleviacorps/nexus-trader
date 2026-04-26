from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class HybridRiskJudgeConfig:
    series_dim: int
    quant_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.10
    num_actions: int = 3


class _ExpGateLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        merged = input_dim + hidden_dim
        self.hidden_dim = int(hidden_dim)
        self.in_proj = nn.Linear(merged, hidden_dim)
        self.forget_proj = nn.Linear(merged, hidden_dim)
        self.candidate_proj = nn.Linear(merged, hidden_dim)
        self.out_proj = nn.Linear(merged, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        merged = torch.cat([x_t, h_prev], dim=-1)
        input_gate = torch.exp(torch.clamp(self.in_proj(merged), min=-6.0, max=3.0))
        forget_gate = torch.exp(torch.clamp(self.forget_proj(merged), min=-6.0, max=3.0))
        candidate = torch.tanh(self.candidate_proj(merged))
        c_next = (forget_gate * c_prev) + (input_gate * candidate)
        h_next = torch.sigmoid(self.out_proj(merged)) * torch.tanh(self.norm(c_next))
        return h_next, c_next


class _xLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = nn.ModuleList(
            [_ExpGateLSTMCell(input_dim if idx == 0 else hidden_dim, hidden_dim) for idx in range(int(num_layers))]
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        batch_size = int(series.size(0))
        output = series
        for layer_index, layer in enumerate(self.layers):
            h = output.new_zeros(batch_size, self.hidden_dim)
            c = output.new_zeros(batch_size, self.hidden_dim)
            states = []
            for step in range(output.size(1)):
                h, c = layer(output[:, step, :], (h, c))
                states.append(h)
            output = torch.stack(states, dim=1)
            if layer_index < len(self.layers) - 1:
                output = self.dropout(output)
        return output[:, -1, :]


class HybridRiskJudge(nn.Module):
    """
    V22 student architecture: xLSTM sequence encoder plus explicit quant risk branch.
    """

    def __init__(self, config: HybridRiskJudgeConfig) -> None:
        super().__init__()
        self.config = config
        self.sequence_encoder = _xLSTMEncoder(
            input_dim=int(config.series_dim),
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
        )
        self.quant_branch = nn.Sequential(
            nn.Linear(int(config.quant_dim), int(config.hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(config.hidden_dim)),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), int(config.hidden_dim)),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(config.hidden_dim) * 2, int(config.hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(config.hidden_dim)),
            nn.Dropout(float(config.dropout)),
        )
        self.action_head = nn.Linear(int(config.hidden_dim), int(config.num_actions))
        self.risk_head = nn.Sequential(nn.Linear(int(config.hidden_dim), 1), nn.Sigmoid())
        self.disagreement_head = nn.Sequential(nn.Linear(int(config.hidden_dim), 1), nn.Sigmoid())

    def forward(self, series: torch.Tensor, quant_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if series.ndim != 3:
            raise ValueError(f"series must be [batch, time, feat], got shape={tuple(series.shape)}")
        if quant_features.ndim != 2:
            raise ValueError(f"quant_features must be [batch, feat], got shape={tuple(quant_features.shape)}")
        sequence_state = self.sequence_encoder(series)
        quant_state = self.quant_branch(quant_features)
        fused = self.fusion(torch.cat([sequence_state, quant_state], dim=-1))
        return {
            "action_logits": self.action_head(fused),
            "risk_pred": self.risk_head(fused).squeeze(-1),
            "disagree_prob": self.disagreement_head(fused).squeeze(-1),
            "embedding": fused,
        }

    @torch.inference_mode()
    def predict(self, series: torch.Tensor, quant_features: torch.Tensor) -> dict[str, Any]:
        outputs = self(series, quant_features)
        probabilities = torch.softmax(outputs["action_logits"], dim=-1)
        action_index = int(torch.argmax(probabilities[0]).item())
        actions = ["BUY", "SELL", "HOLD"]
        return {
            "action": actions[action_index],
            "confidence": float(probabilities[0, action_index].item()),
            "probabilities": probabilities[0].detach().cpu(),
            "risk_pred": float(outputs["risk_pred"][0].item()),
            "disagree_prob": float(outputs["disagree_prob"][0].item()),
        }


__all__ = ["HybridRiskJudge", "HybridRiskJudgeConfig"]
