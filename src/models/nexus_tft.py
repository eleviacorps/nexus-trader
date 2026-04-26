from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from config.project_config import FEATURE_DIM_TOTAL

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class NexusTFTConfig:
    input_dim: int = FEATURE_DIM_TOTAL
    hidden_dim: int = 128
    lstm_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 1
    regime_count: int = 4
    router_hidden_dim: int = 64
    router_temperature: float = 1.0


def expand_feature_vector(values: Sequence[float], new_dim: int, fill_value: float = 0.0) -> List[float]:
    output = list(values[:new_dim])
    if len(output) < new_dim:
        output.extend([fill_value] * (new_dim - len(output)))
    return output


def expand_feature_matrix_columns(
    matrix: Sequence[Sequence[float]],
    old_input_dim: int,
    new_input_dim: int,
    fill_value: float = 0.0,
) -> List[List[float]]:
    expanded: List[List[float]] = []
    for row in matrix:
        prefix = list(row[:old_input_dim])
        if len(prefix) < old_input_dim:
            prefix.extend([0.0] * (old_input_dim - len(prefix)))
        prefix.extend([fill_value] * max(0, new_input_dim - old_input_dim))
        expanded.append(prefix[:new_input_dim])
    return expanded


if nn is not None:
    class VariableSelectionNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.gate = nn.Linear(input_dim, input_dim)
            self.input_projection = nn.Linear(input_dim, hidden_dim)

        def forward(self, x):
            weights = torch.softmax(self.gate(x), dim=-1)
            projected = self.input_projection(x * weights)
            return projected, weights


    class ExpertHead(nn.Module):
        def __init__(self, hidden_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, max(hidden_dim // 2, 32)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(hidden_dim // 2, 32), output_dim),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)


    class NexusTFT(nn.Module):
        def __init__(self, config: NexusTFTConfig | None = None):
            super().__init__()
            self.config = config or NexusTFTConfig()
            self.vsn = VariableSelectionNetwork(self.config.input_dim, self.config.hidden_dim)
            self.encoder = nn.LSTM(
                input_size=self.config.hidden_dim,
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.lstm_layers,
                dropout=self.config.dropout,
                batch_first=True,
            )
            self.attention = nn.Sequential(
                nn.Linear(self.config.hidden_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            self.base_head = nn.Sequential(
                nn.LayerNorm(self.config.hidden_dim),
                nn.Linear(self.config.hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(64, self.config.output_dim),
                nn.Sigmoid(),
            )
            self.regime_router = nn.Sequential(
                nn.LayerNorm(self.config.hidden_dim),
                nn.Linear(self.config.hidden_dim, self.config.router_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.router_hidden_dim, self.config.regime_count),
            )
            self.regime_embeddings = nn.Embedding(self.config.regime_count, self.config.hidden_dim)
            self.expert_heads = nn.ModuleList(
                [
                    ExpertHead(
                        hidden_dim=self.config.hidden_dim,
                        output_dim=self.config.output_dim,
                        dropout=self.config.dropout,
                    )
                    for _ in range(self.config.regime_count)
                ]
            )
            self.regime_residual = nn.Sequential(
                nn.LayerNorm(self.config.hidden_dim),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )

        def optimizer_groups(self, old_layers_lr: float, new_layers_lr: float):
            inherited_params = list(self.vsn.parameters()) + list(self.encoder.parameters()) + list(self.attention.parameters())
            new_params = list(self.base_head.parameters()) + list(self.regime_router.parameters()) + list(self.regime_embeddings.parameters()) + list(self.expert_heads.parameters()) + list(self.regime_residual.parameters())
            return [
                {"params": inherited_params, "lr": old_layers_lr},
                {"params": new_params, "lr": new_layers_lr},
            ]

        def forward(self, x, return_feature_importance: bool = False, return_diagnostics: bool = False):
            encoded, importance = self.vsn(x)
            outputs, _ = self.encoder(encoded)
            weights = torch.softmax(self.attention(outputs), dim=1)
            context = (outputs * weights).sum(dim=1)
            router_logits = self.regime_router(context) / max(self.config.router_temperature, 1e-6)
            regime_probabilities = torch.softmax(router_logits, dim=-1)
            regime_context = regime_probabilities @ self.regime_embeddings.weight
            mixed_context = context + self.regime_residual(regime_context)
            expert_predictions = torch.stack([head(mixed_context) for head in self.expert_heads], dim=1)
            routed_prediction = (expert_predictions * regime_probabilities.unsqueeze(-1)).sum(dim=1)
            base_prediction = self.base_head(mixed_context)
            regime_confidence = regime_probabilities.max(dim=-1).values.unsqueeze(-1)
            blend_weight = 0.55 + (0.25 * regime_confidence)
            prediction = (blend_weight * routed_prediction) + ((1.0 - blend_weight) * base_prediction)
            if self.config.output_dim == 1:
                prediction = prediction.squeeze(-1)
                routed_prediction = routed_prediction.squeeze(-1)
                base_prediction = base_prediction.squeeze(-1)
            diagnostics = {
                "regime_probabilities": regime_probabilities,
                "regime_confidence": regime_confidence.squeeze(-1),
                "routed_prediction": routed_prediction,
                "base_prediction": base_prediction,
                "expert_predictions": expert_predictions.squeeze(-1) if self.config.output_dim == 1 else expert_predictions,
            }
            if return_feature_importance and return_diagnostics:
                return prediction, importance.mean(dim=1), diagnostics
            if return_feature_importance:
                return prediction, importance.mean(dim=1)
            if return_diagnostics:
                return prediction, diagnostics
            return prediction
else:  # pragma: no cover
    class NexusTFT:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for NexusTFT.")


def _torch_xavier_fill(rows: int, cols: int):
    values = torch.empty(rows, cols)
    torch.nn.init.xavier_uniform_(values)
    return values


def expand_tensor_columns(tensor: Any, new_cols: int):
    rows, old_cols = tensor.shape
    expanded = _torch_xavier_fill(rows, new_cols)
    expanded[:, :old_cols] = tensor
    return expanded


def expand_tensor_rows(tensor: Any, new_rows: int):
    old_rows, cols = tensor.shape
    expanded = _torch_xavier_fill(new_rows, cols)
    expanded[:old_rows, :] = tensor
    return expanded


def expand_tensor_vector(tensor: Any, new_size: int):
    expanded = tensor.new_zeros(new_size)
    expanded[: tensor.shape[0]] = tensor
    return expanded


def migrate_legacy_state_dict(
    legacy_state: Mapping[str, Any],
    model_state: Mapping[str, Any],
    old_input_dim: int = 36,
    new_input_dim: int = FEATURE_DIM_TOTAL,
) -> Dict[str, Any]:
    migrated: Dict[str, Any] = {}
    for key, target_value in model_state.items():
        source_value = legacy_state.get(key)
        if source_value is None:
            migrated[key] = target_value
            continue
        if torch is not None and hasattr(source_value, "shape") and hasattr(target_value, "shape"):
            if tuple(source_value.shape) == tuple(target_value.shape):
                migrated[key] = source_value
                continue
            if len(target_value.shape) == 2 and len(source_value.shape) == 2:
                if source_value.shape[1] == old_input_dim and target_value.shape[1] == new_input_dim:
                    migrated[key] = expand_tensor_columns(source_value, target_value.shape[1])
                    continue
                if source_value.shape[0] == old_input_dim and target_value.shape[0] == new_input_dim:
                    migrated[key] = expand_tensor_rows(source_value, target_value.shape[0])
                    continue
            if len(target_value.shape) == 1 and len(source_value.shape) == 1:
                if source_value.shape[0] == old_input_dim and target_value.shape[0] == new_input_dim:
                    migrated[key] = expand_tensor_vector(source_value, target_value.shape[0])
                    continue
        migrated[key] = target_value
    return migrated


def load_checkpoint_with_expansion(
    model: Any,
    checkpoint_path: Path,
    old_input_dim: int = 36,
    new_input_dim: int = FEATURE_DIM_TOTAL,
):
    if torch is None:
        raise ImportError("PyTorch is required to load checkpoints.")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(checkpoint_path, map_location="cpu")
    legacy_state = payload.get("model_state_dict") or payload.get("state_dict") or payload
    migrated_state = migrate_legacy_state_dict(legacy_state, model.state_dict(), old_input_dim, new_input_dim)
    missing, unexpected = model.load_state_dict(migrated_state, strict=False)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def summarize_feature_importance(feature_names: Sequence[str], importance_rows: Sequence[Sequence[float]]) -> Dict[str, float]:
    totals = {name: 0.0 for name in feature_names}
    count = 0
    for row in importance_rows:
        count += 1
        for name, value in zip(feature_names, row):
            totals[name] += float(value)
    if count == 0:
        return totals
    return {name: value / count for name, value in totals.items()}
