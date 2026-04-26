from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config.project_config import V14_SSC_MODEL_PATH


class SimulationCritic(nn.Module):
    def __init__(self, branch_dim: int, context_dim: int, hidden: int = 64):
        super().__init__()
        self.branch_proj = nn.Sequential(
            nn.Linear(branch_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.interaction = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.output_head = nn.Linear(hidden, 3)

    def forward(self, branch_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        b = self.branch_proj(branch_features)
        c = self.context_proj(context_features)
        combined = self.interaction(torch.cat([b, c], dim=-1))
        return torch.sigmoid(self.output_head(combined))

    def critique_score(self, branch_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        out = self.forward(branch_features, context_features)
        assumption_risk = out[:, 0]
        context_consistency = out[:, 1]
        contradiction_depth = out[:, 2]
        return (
            0.40 * context_consistency
            + 0.35 * (1.0 - assumption_risk)
            + 0.25 * contradiction_depth
        )


def build_ssc_labels(archive_df: pd.DataFrame) -> pd.DataFrame:
    working = archive_df.copy()
    direction = np.sign(pd.to_numeric(working.get("branch_direction", 0.0), errors="coerce").fillna(0.0))
    momentum = np.sign(pd.to_numeric(working.get("bcfe_macd_hist", 0.0), errors="coerce").fillna(0.0))
    grouped = working.groupby(["dominant_regime", "branch_label"], sort=False)["setl_target_net_unit_pnl"]
    normalized_std = grouped.transform("std").fillna(0.0)
    if float(normalized_std.max()) > 0.0:
        normalized_std = normalized_std / float(normalized_std.max())
    working["ssc_assumption_risk"] = np.clip(normalized_std.to_numpy(dtype=np.float32), 0.0, 1.0)
    consistency = 0.5 + 0.5 * (direction.to_numpy(dtype=np.float32) * momentum.to_numpy(dtype=np.float32))
    working["ssc_context_consistency"] = np.clip(consistency, 0.0, 1.0)

    anchor_series = pd.to_numeric(working.get("anchor_price", 0.0), errors="coerce").fillna(0.0)
    p5_series = pd.to_numeric(working.get("predicted_price_5m", anchor_series), errors="coerce").fillna(anchor_series)
    p10_series = pd.to_numeric(working.get("predicted_price_10m", p5_series), errors="coerce").fillna(p5_series)
    p15_series = pd.to_numeric(working.get("predicted_price_15m", p10_series), errors="coerce").fillna(p10_series)
    anchor = anchor_series.to_numpy(dtype=np.float32)
    p5 = p5_series.to_numpy(dtype=np.float32)
    p10 = p10_series.to_numpy(dtype=np.float32)
    p15 = p15_series.to_numpy(dtype=np.float32)
    depth_values: list[float] = []
    for first, second, third, fourth, base_direction in zip(anchor, p5, p10, p15, direction.tolist(), strict=False):
        path = np.asarray([first, second, third, fourth], dtype=np.float32)
        deltas = np.diff(path)
        sign = int(base_direction) or 1
        contradiction_idx = next((idx for idx, delta in enumerate(deltas, start=1) if int(np.sign(delta) or sign) != sign), len(deltas))
        depth_values.append(float(contradiction_idx / max(len(deltas), 1)))
    working["ssc_contradiction_depth"] = np.clip(np.asarray(depth_values, dtype=np.float32), 0.0, 1.0)
    return working


def _tensor_frame(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame[list(columns)].fillna(0.0).to_numpy(dtype=np.float32)


def train_ssc_model(
    frame: pd.DataFrame,
    *,
    branch_feature_names: Sequence[str],
    context_feature_names: Sequence[str],
    device: str | None = None,
    epochs: int = 18,
    batch_size: int = 512,
    lr: float = 3e-4,
    checkpoint_path: Path = V14_SSC_MODEL_PATH,
) -> dict[str, Any]:
    working = build_ssc_labels(frame)
    branch_values = _tensor_frame(working, branch_feature_names)
    context_values = _tensor_frame(working, context_feature_names)
    labels = working[["ssc_assumption_risk", "ssc_context_consistency", "ssc_contradiction_depth"]].to_numpy(dtype=np.float32)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = TensorDataset(
        torch.from_numpy(branch_values),
        torch.from_numpy(context_values),
        torch.from_numpy(labels),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = SimulationCritic(branch_dim=len(branch_feature_names), context_dim=len(context_feature_names)).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[float] = []

    for _ in range(max(int(epochs), 1)):
        model.train()
        running = 0.0
        steps = 0
        for branch_batch, context_batch, label_batch in loader:
            branch_batch = branch_batch.to(target_device)
            context_batch = context_batch.to(target_device)
            label_batch = label_batch.to(target_device)
            pred = model(branch_batch, context_batch)
            loss = F.mse_loss(pred, label_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            steps += 1
        history.append(running / max(steps, 1))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "branch_feature_names": tuple(branch_feature_names),
            "context_feature_names": tuple(context_feature_names),
            "history": [float(value) for value in history],
        },
        checkpoint_path,
    )
    return {"model": model, "history": history, "device": str(target_device)}


def load_ssc_model(path: Path = V14_SSC_MODEL_PATH, *, map_location: str | None = None) -> tuple[SimulationCritic, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    payload = torch.load(path, map_location=map_location or "cpu")
    branch_feature_names = tuple(payload["branch_feature_names"])
    context_feature_names = tuple(payload["context_feature_names"])
    model = SimulationCritic(branch_dim=len(branch_feature_names), context_dim=len(context_feature_names))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, branch_feature_names, context_feature_names, payload


def score_ssc(
    model: SimulationCritic,
    frame: pd.DataFrame,
    *,
    branch_feature_names: Sequence[str],
    context_feature_names: Sequence[str],
    device: str | None = None,
) -> np.ndarray:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    model.eval()
    branch_tensor = torch.from_numpy(_tensor_frame(frame, branch_feature_names)).to(target_device)
    context_tensor = torch.from_numpy(_tensor_frame(frame, context_feature_names)).to(target_device)
    with torch.no_grad():
        scores = model.critique_score(branch_tensor, context_tensor).detach().cpu().numpy().astype(np.float32)
    return scores


def evaluate_ssc(
    model: SimulationCritic,
    frame: pd.DataFrame,
    *,
    branch_feature_names: Sequence[str],
    context_feature_names: Sequence[str],
    device: str | None = None,
) -> dict[str, Any]:
    labeled = build_ssc_labels(frame)
    target = labeled[["ssc_assumption_risk", "ssc_context_consistency", "ssc_contradiction_depth"]].to_numpy(dtype=np.float32)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(_tensor_frame(labeled, branch_feature_names)).to(target_device),
            torch.from_numpy(_tensor_frame(labeled, context_feature_names)).to(target_device),
        ).detach().cpu().numpy().astype(np.float32)
    mae = np.mean(np.abs(pred - target), axis=0)
    return {
        "assumption_risk_mae": float(mae[0]),
        "context_consistency_mae": float(mae[1]),
        "contradiction_depth_mae": float(mae[2]),
        "composite_score_mean": float(score_ssc(model, labeled, branch_feature_names=branch_feature_names, context_feature_names=context_feature_names, device=device).mean()),
    }
