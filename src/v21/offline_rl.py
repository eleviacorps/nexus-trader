from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ConservativeQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


@dataclass(frozen=True)
class CQLBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


def conservative_q_loss(
    q_network: ConservativeQNetwork,
    target_network: ConservativeQNetwork,
    batch: CQLBatch,
    *,
    gamma: float = 0.99,
    alpha: float = 1.0,
) -> torch.Tensor:
    q_values = q_network(batch.states)
    chosen_q = q_values.gather(1, batch.actions.long().unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q = target_network(batch.next_states).max(dim=1).values
        target = batch.rewards + (1.0 - batch.dones.float()) * float(gamma) * next_q
    td_loss = torch.mean((chosen_q - target) ** 2)
    conservative_penalty = torch.logsumexp(q_values, dim=1).mean() - chosen_q.mean()
    return td_loss + float(alpha) * conservative_penalty


__all__ = ["ConservativeQNetwork", "CQLBatch", "conservative_q_loss"]
