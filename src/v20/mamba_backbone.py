from __future__ import annotations

import torch
from torch import nn


class NexusBiMamba(nn.Module):
    def __init__(self, n_features: int = 350, d_model: int = 256, n_layers: int = 4, sequence_len: int = 240) -> None:
        super().__init__()
        self.sequence_len = int(sequence_len)
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dropout=0.1)
        self.forward_ssm = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=1) for _ in range(n_layers)])
        self.backward_ssm = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=1) for _ in range(n_layers)])
        self.graph_adj = nn.Parameter(torch.eye(n_features))
        self.graph_conv = nn.Linear(n_features, n_features)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head_dir_15m = nn.Linear(d_model, 1)
        self.head_dir_30m = nn.Linear(d_model, 1)
        self.head_vol_env = nn.Linear(d_model, 3)
        self.head_regime = nn.Linear(d_model, 6)
        self.head_range = nn.Linear(d_model, 3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        adj_norm = torch.softmax(self.graph_adj, dim=-1)
        x_graph = torch.einsum("bij,jk->bik", x, adj_norm)
        x = x + self.graph_conv(x_graph)
        x_proj = self.input_proj(x)
        fwd = x_proj
        bwd = torch.flip(x_proj, dims=[1])
        for f_layer, b_layer in zip(self.forward_ssm, self.backward_ssm, strict=False):
            fwd = f_layer(fwd)
            bwd = b_layer(bwd)
        bwd = torch.flip(bwd, dims=[1])
        return self.norm(self.fusion(torch.cat([fwd, bwd], dim=-1)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        h_last = h[:, -1, :]
        return {
            "dir_15m": self.head_dir_15m(h_last).squeeze(-1),
            "dir_30m": self.head_dir_30m(h_last).squeeze(-1),
            "vol_env": self.head_vol_env(h_last),
            "regime": self.head_regime(h_last),
            "range": self.head_range(h_last),
            "hidden": h_last,
        }
