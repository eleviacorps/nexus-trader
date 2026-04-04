from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.v9.branch_features_v9 import BRANCH_FEATURES_V9


@dataclass(frozen=True)
class SelectorTorchReport:
    device: str
    feature_count: int
    epochs: int
    train_loss: float
    validation_loss: float
    top1_accuracy: float
    top3_containment: float
    event_win_rate: float


class BranchSelectorTorch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, inputs.shape[-1])
        scores = self.network(flat).reshape(*original_shape)
        return scores


def _resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_group_tensors(frame, feature_names: Sequence[str], target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sorted_frame = frame.sort_values(["sample_id", "branch_id"]).reset_index(drop=True)
    sample_groups = []
    max_branches = 0
    for _, sample in sorted_frame.groupby("sample_id", sort=False):
        group_features = sample[list(feature_names)].to_numpy(dtype=np.float32)
        group_targets = sample[target_col].to_numpy(dtype=np.float32)
        if float(group_targets.sum()) <= 0.0:
            group_targets = sample["composite_winner_label"].to_numpy(dtype=np.float32)
        if float(group_targets.sum()) <= 0.0:
            group_targets = np.ones_like(group_targets, dtype=np.float32)
        group_targets = group_targets / max(float(group_targets.sum()), 1e-6)
        group_event_wins = (
            (np.sign(sample["actual_final_return"].to_numpy(dtype=np.float32)) == np.sign(sample["branch_direction"].to_numpy(dtype=np.float32))).astype(np.float32)
        )
        sample_groups.append((group_features, group_targets, group_event_wins))
        max_branches = max(max_branches, group_features.shape[0])
    groups = []
    targets = []
    event_wins = []
    masks = []
    for group_features, group_targets, group_event_wins in sample_groups:
        branch_count = group_features.shape[0]
        feature_pad = np.zeros((max_branches, len(feature_names)), dtype=np.float32)
        target_pad = np.zeros(max_branches, dtype=np.float32)
        event_pad = np.zeros(max_branches, dtype=np.float32)
        mask = np.zeros(max_branches, dtype=np.float32)
        feature_pad[:branch_count] = group_features
        target_pad[:branch_count] = group_targets
        event_pad[:branch_count] = group_event_wins
        mask[:branch_count] = 1.0
        groups.append(feature_pad)
        targets.append(target_pad)
        event_wins.append(event_pad)
        masks.append(mask)
    return (
        np.stack(groups).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.stack(event_wins).astype(np.float32),
        np.stack(masks).astype(np.float32),
    )


def _split_train_validation(group_count: int, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    group_count = max(group_count, 1)
    validation_count = min(max(int(group_count * validation_fraction), 1), max(group_count - 1, 1))
    indices = np.arange(group_count, dtype=np.int64)
    split_point = group_count - validation_count
    return indices[:split_point], indices[split_point:]


def _evaluate_predictions(scores: np.ndarray, targets: np.ndarray, event_wins: np.ndarray, masks: np.ndarray) -> tuple[float, float, float]:
    top1 = []
    top3 = []
    event_win = []
    for group_scores, group_targets, group_event, group_mask in zip(scores, targets, event_wins, masks, strict=False):
        valid = np.flatnonzero(group_mask > 0.0)
        if valid.size == 0:
            continue
        ranking = valid[np.argsort(group_scores[valid])[::-1]]
        best_index = int(np.argmax(group_targets))
        top1.append(float(ranking[0] == best_index))
        top3.append(float(best_index in set(ranking[:3].tolist())))
        event_win.append(float(group_event[ranking[0]]))
    return float(np.mean(top1)), float(np.mean(top3)), float(np.mean(event_win))


def train_selector_torch(
    frame,
    *,
    feature_names: Sequence[str] = BRANCH_FEATURES_V9,
    target_col: str = "composite_score",
    epochs: int = 12,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    device: str | None = None,
    seed: int = 42,
) -> tuple[BranchSelectorTorch, SelectorTorchReport]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x_groups, y_groups, event_groups, mask_groups = _build_group_tensors(frame, feature_names=feature_names, target_col=target_col)
    train_idx, valid_idx = _split_train_validation(len(x_groups), validation_fraction)
    torch_device = _resolve_device(device)

    model = BranchSelectorTorch(input_dim=len(feature_names)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_x = torch.from_numpy(x_groups[train_idx]).to(torch_device)
    train_y = torch.from_numpy(y_groups[train_idx]).to(torch_device)
    train_mask = torch.from_numpy(mask_groups[train_idx]).to(torch_device)
    valid_x = torch.from_numpy(x_groups[valid_idx]).to(torch_device)
    valid_y = torch.from_numpy(y_groups[valid_idx]).to(torch_device)
    valid_mask = torch.from_numpy(mask_groups[valid_idx]).to(torch_device)

    final_train_loss = 0.0
    final_valid_loss = 0.0
    for _ in range(max(int(epochs), 1)):
        model.train()
        order = torch.randperm(train_x.shape[0], device=torch_device)
        batch_losses = []
        for start in range(0, train_x.shape[0], max(int(batch_size), 1)):
            batch_index = order[start : start + batch_size]
            batch_x = train_x[batch_index]
            batch_y = train_y[batch_index]
            batch_mask = train_mask[batch_index]
            logits = model(batch_x).masked_fill(batch_mask <= 0.0, -1e9)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(batch_y * log_probs).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_x).masked_fill(valid_mask <= 0.0, -1e9)
            final_valid_loss = float((-(valid_y * torch.log_softmax(valid_logits, dim=1)).sum(dim=1).mean()).detach().cpu())

    model.eval()
    with torch.no_grad():
        valid_scores = model(valid_x).masked_fill(valid_mask <= 0.0, -1e9).detach().cpu().numpy().astype(np.float32)
    top1_accuracy, top3_containment, event_win_rate = _evaluate_predictions(
        valid_scores,
        y_groups[valid_idx],
        event_groups[valid_idx],
        mask_groups[valid_idx],
    )
    report = SelectorTorchReport(
        device=str(torch_device),
        feature_count=len(feature_names),
        epochs=max(int(epochs), 1),
        train_loss=final_train_loss,
        validation_loss=final_valid_loss,
        top1_accuracy=top1_accuracy,
        top3_containment=top3_containment,
        event_win_rate=event_win_rate,
    )
    return model, report


def save_selector_torch(model: BranchSelectorTorch, path: Path, feature_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": list(feature_names),
            "input_dim": len(feature_names),
        },
        path,
    )


def load_selector_torch(path: Path, device: str | None = None) -> tuple[BranchSelectorTorch, list[str]]:
    payload = torch.load(path, map_location=_resolve_device(device))
    feature_names = list(payload.get("feature_names", list(BRANCH_FEATURES_V9)))
    model = BranchSelectorTorch(input_dim=int(payload.get("input_dim", len(feature_names))))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, feature_names


def score_selector_torch(model: BranchSelectorTorch, frame, feature_names: Sequence[str], device: str | None = None) -> np.ndarray:
    torch_device = _resolve_device(device)
    model = model.to(torch_device)
    inputs = torch.from_numpy(np.array(frame[list(feature_names)].to_numpy(dtype=np.float32), copy=True)).to(torch_device)
    with torch.no_grad():
        scores = model(inputs.unsqueeze(1)).squeeze(1).squeeze(-1)
    return scores.detach().cpu().numpy().astype(np.float32)
