from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LeeOscillatorConfig:
    frequency: float
    damping: float
    coupling: float
    nonlinearity: float


LEE_OSCILLATOR_CONFIGS: dict[str, LeeOscillatorConfig] = {
    "T9": LeeOscillatorConfig(frequency=0.92, damping=0.84, coupling=0.28, nonlinearity=0.42),
}


class LeeCOC(nn.Module):
    def __init__(self, oscillator_type: str = "T9", n_steps: int = 9) -> None:
        super().__init__()
        self.oscillator_type = oscillator_type
        self.n_steps = int(max(n_steps, 2))
        self.config = LEE_OSCILLATOR_CONFIGS.get(oscillator_type, LEE_OSCILLATOR_CONFIGS["T9"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        trajectory = self._run_oscillator(x)
        return torch.max(trajectory, dim=0).values

    def _run_oscillator(self, x: torch.Tensor) -> torch.Tensor:
        config = self.config
        state = x
        velocity = torch.zeros_like(x)
        points = []
        for step in range(self.n_steps):
            phase = float(step + 1)
            chaotic_drive = (
                torch.sin(state * (config.frequency * phase))
                + config.nonlinearity * torch.tanh(state)
                + 0.25 * torch.cos(state * (phase * 0.37))
            )
            velocity = (config.damping * velocity) + chaotic_drive - (config.coupling * state)
            state = state + velocity
            points.append(state.unsqueeze(0))
        return torch.cat(points, dim=0)
