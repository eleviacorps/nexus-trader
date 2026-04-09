from __future__ import annotations

import sys

import torch
from torch import nn


class VariableSelectionNetwork(nn.Module):
    """Regime-conditioned per-timestep feature weighting."""

    def __init__(self, n_features: int, d_model: int, n_regimes: int = 6) -> None:
        super().__init__()
        self.regime_embed = nn.Embedding(int(n_regimes), 32)
        self.grn = nn.Sequential(
            nn.Linear(int(n_features) + 32, int(d_model)),
            nn.ELU(),
            nn.Linear(int(d_model), int(n_features)),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, regime_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        regime_emb = self.regime_embed(regime_ids)
        combined = torch.cat([x, regime_emb], dim=-1)
        weights = self.softmax(self.grn(combined))
        return x * weights, weights


class sLSTMBlock(nn.Module):
    """Scalar xLSTM block with stabilized exponential gating."""

    def __init__(self, d_model: int, d_state: int = 64) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.in_norm = nn.LayerNorm(self.d_model)
        self.W_i = nn.Linear(self.d_model, self.d_state)
        self.W_f = nn.Linear(self.d_model, self.d_state)
        self.W_z = nn.Linear(self.d_model, self.d_state)
        self.W_o = nn.Linear(self.d_model, self.d_state)
        self.proj_out = nn.Linear(self.d_state, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        x = self.in_norm(x)
        h = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)
        c = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)
        m = torch.zeros(batch, self.d_state, device=x.device, dtype=x.dtype)
        outputs: list[torch.Tensor] = []

        for t in range(seq):
            xt = x[:, t, :]
            log_i = self.W_i(xt).clamp(min=-20.0, max=20.0)
            log_f = self.W_f(xt).clamp(min=-20.0, max=20.0)
            z = torch.tanh(self.W_z(xt))
            o = torch.sigmoid(self.W_o(xt))

            m_new = torch.maximum(m + log_f, log_i)
            i_stable = torch.exp(log_i - m_new)
            f_stable = torch.exp(m + log_f - m_new)
            m = m_new

            c = f_stable * c + i_stable * z
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(1))

        return self.proj_out(torch.cat(outputs, dim=1))


class NexusXLSTM(nn.Module):
    """VSN + stacked xLSTM blocks with multitask prediction heads."""

    def __init__(self, n_features: int = 350, d_model: int = 512, n_layers: int = 4, n_regimes: int = 6) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_regimes = int(n_regimes)
        self.vsn = VariableSelectionNetwork(self.n_features, int(d_model), self.n_regimes)
        self.input_proj = nn.Linear(self.n_features, int(d_model))
        self.xlstm_layers = nn.ModuleList([sLSTMBlock(int(d_model)) for _ in range(int(n_layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(int(d_model)) for _ in range(int(n_layers))])
        self.head_dir_15m = nn.Linear(int(d_model), 1)
        self.head_dir_30m = nn.Linear(int(d_model), 1)
        self.head_vol_env = nn.Linear(int(d_model), 3)
        self.head_regime = nn.Linear(int(d_model), self.n_regimes)
        self.head_range = nn.Linear(int(d_model), 3)

    def encode(self, x: torch.Tensor, regime_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_weighted, vsn_weights = self.vsn(x, regime_ids)
        h = self.input_proj(x_weighted)
        for layer, norm in zip(self.xlstm_layers, self.norms, strict=False):
            h = norm(h + layer(h))
        return h, vsn_weights

    def forward(self, x: torch.Tensor, regime_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        h, vsn_weights = self.encode(x, regime_ids)
        h_last = h[:, -1, :]
        return {
            "dir_15m": self.head_dir_15m(h_last).squeeze(-1),
            "dir_30m": self.head_dir_30m(h_last).squeeze(-1),
            "vol_env": self.head_vol_env(h_last),
            "regime": self.head_regime(h_last),
            "range": self.head_range(h_last),
            "hidden": h_last,
            "vsn_weights": vsn_weights[:, -1, :],
        }


sys.modules.setdefault("src.v21.xLSTM_backbone", sys.modules[__name__])

__all__ = ["VariableSelectionNetwork", "sLSTMBlock", "NexusXLSTM"]
