from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    patience: int = 3
    batch_size: int = 512
    inherited_lr: float = 1e-4
    new_layers_lr: float = 5e-4


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sim_weight_from_confidence(confidence: float) -> float:
    return clamp((confidence - 0.3) / 0.7, 0.0, 1.0)


def combined_loss_weights(confidence: float) -> Dict[str, float]:
    sim_weight = sim_weight_from_confidence(confidence)
    denominator = 3.0 + sim_weight
    return {
        "real_weight": 3.0 / denominator,
        "sim_weight": sim_weight / denominator,
    }


def save_feature_importance_report(path: Path, report: Mapping[str, float]) -> None:
    path.write_text(json.dumps(dict(report), indent=2), encoding="utf-8")


def save_training_config(path: Path, config: TrainingConfig) -> None:
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def build_optimizer(model: Any, old_layers_lr: float = 1e-4, new_layers_lr: float = 5e-4):
    if torch is None:
        raise ImportError("PyTorch is required to build the optimizer.")
    if hasattr(model, "optimizer_groups"):
        return torch.optim.AdamW(model.optimizer_groups(old_layers_lr, new_layers_lr))
    return torch.optim.AdamW(model.parameters(), lr=old_layers_lr)
