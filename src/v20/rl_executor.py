from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


class MacroStyleHyperAgent(nn.Module):
    def __init__(self, n_regimes: int = 6, memory_len: int = 20, d_macro: int = 8) -> None:
        super().__init__()
        self.n_regimes = int(n_regimes)
        self.memory_len = int(memory_len)
        self.memory_encoder = nn.LSTM(
            input_size=n_regimes + 2,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(n_regimes + 64 + d_macro, 128),
            nn.GELU(),
            nn.Linear(128, n_regimes),
        )

    def forward(self, regime_probs: torch.Tensor, memory_sequence: torch.Tensor, macro_state: torch.Tensor) -> torch.Tensor:
        memory_h, _ = self.memory_encoder(memory_sequence)
        memory_last = memory_h[:, -1, :]
        context = torch.cat([regime_probs, memory_last, macro_state], dim=-1)
        return torch.softmax(self.context_fusion(context), dim=-1)


@dataclass
class RegimeSubAgent:
    name: str
    bias: float
    volatility_penalty: float

    def score(self, direction_signal: float, confidence: float, macro_bias: float, volatility: float) -> float:
        score = (self.bias * direction_signal) + (0.9 * confidence * np.sign(direction_signal or 1.0)) + (0.35 * macro_bias)
        score -= self.volatility_penalty * abs(volatility)
        return float(score)


@dataclass
class ExecutorDecision:
    action: str
    lot_multiplier: float
    hyper_weights: dict[str, float]
    active_sub_agent: str
    expected_reward: float


@dataclass
class HierarchicalExecutor:
    memory_len: int = 20
    memory: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hyper_agent = MacroStyleHyperAgent()
        self.sub_agents = [
            RegimeSubAgent("low_vol_range", bias=0.20, volatility_penalty=0.15),
            RegimeSubAgent("trending_up", bias=1.00, volatility_penalty=0.10),
            RegimeSubAgent("trending_down", bias=-1.00, volatility_penalty=0.10),
            RegimeSubAgent("breakout", bias=0.70, volatility_penalty=0.05),
            RegimeSubAgent("mean_revert", bias=-0.35, volatility_penalty=0.12),
            RegimeSubAgent("panic", bias=-0.55, volatility_penalty=0.30),
        ]

    def _memory_tensor(self) -> torch.Tensor:
        rows = self.memory[-self.memory_len :]
        if not rows:
            rows = [[0.0] * 8]
        while len(rows) < self.memory_len:
            rows.insert(0, [0.0] * len(rows[0]))
        return torch.tensor([rows], dtype=torch.float32)

    def decide(
        self,
        *,
        regime_probs: list[float],
        direction_signal: float,
        confidence: float,
        macro_state: list[float],
        volatility: float,
        kelly_fraction: float,
        branch_rewards: list[float],
    ) -> ExecutorDecision:
        regime_vector = np.asarray(regime_probs[:6], dtype=np.float32)
        if regime_vector.size < 6:
            regime_vector = np.pad(regime_vector, (0, 6 - regime_vector.size))
        macro_vector = np.asarray((macro_state + [0.0] * 8)[:8], dtype=np.float32)
        with torch.no_grad():
            weights = self.hyper_agent(
                torch.tensor([regime_vector], dtype=torch.float32),
                self._memory_tensor(),
                torch.tensor([macro_vector], dtype=torch.float32),
            )[0].cpu().numpy()
        scores = np.asarray(
            [
                sub_agent.score(direction_signal, confidence, macro_vector[0], volatility)
                for sub_agent in self.sub_agents
            ],
            dtype=np.float64,
        )
        combined = float(np.dot(weights, scores))
        action = "BUY" if combined > 0.10 else "SELL" if combined < -0.10 else "HOLD"
        expected_reward = float(np.mean(branch_rewards)) if branch_rewards else 0.0
        active_index = int(np.argmax(weights))
        lot_multiplier = float(np.clip(kelly_fraction * max(confidence, 0.1), 0.0, 1.0))
        return ExecutorDecision(
            action=action,
            lot_multiplier=lot_multiplier,
            hyper_weights={self.sub_agents[idx].name: round(float(value), 6) for idx, value in enumerate(weights.tolist())},
            active_sub_agent=self.sub_agents[active_index].name,
            expected_reward=round(expected_reward, 6),
        )

    def record_outcome(self, *, regime_probs: list[float], action: str, reward: float) -> None:
        signal = 1.0 if action == "BUY" else -1.0 if action == "SELL" else 0.0
        row = [float(value) for value in (regime_probs[:6] + [signal, float(reward)])]
        self.memory.append(row)
        self.memory = self.memory[-self.memory_len :]
