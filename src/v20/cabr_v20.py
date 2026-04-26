from __future__ import annotations

import numpy as np
import pandas as pd
from torch import nn


class CABR_V20(nn.Module):
    def __init__(self, d_branch: int = 64, d_context: int = 128, d_model: int = 256, n_heads: int = 8, n_layers: int = 4) -> None:
        super().__init__()
        self.branch_mixer = nn.Sequential(
            nn.Linear(d_branch, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.context_proj = nn.Linear(d_context, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regime_heads = nn.ModuleDict(
            {
                "low_vol_range": nn.Linear(d_model, 1),
                "trending_up": nn.Linear(d_model, 1),
                "trending_down": nn.Linear(d_model, 1),
                "breakout": nn.Linear(d_model, 1),
                "mean_revert": nn.Linear(d_model, 1),
                "panic": nn.Linear(d_model, 1),
            }
        )
        self.global_head = nn.Linear(d_model, 1)


def heuristic_branch_scores(candidate_frame: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_frame.copy()
    current_price = pd.to_numeric(frame.get("anchor_price"), errors="coerce").replace(0.0, np.nan).ffill().bfill()
    predicted = pd.to_numeric(frame.get("predicted_price_15m"), errors="coerce").fillna(current_price)
    move = (predicted - current_price) / current_price.replace(0.0, np.nan)
    base = (
        0.20 * pd.to_numeric(frame.get("generator_probability"), errors="coerce").fillna(0.5)
        + 0.14 * pd.to_numeric(frame.get("analog_similarity"), errors="coerce").fillna(0.5)
        + 0.12 * pd.to_numeric(frame.get("quant_regime_strength"), errors="coerce").fillna(0.5)
        + 0.12 * pd.to_numeric(frame.get("quant_route_confidence"), errors="coerce").fillna(0.5)
        + 0.10 * pd.to_numeric(frame.get("branch_confidence"), errors="coerce").fillna(0.5)
        + 0.10 * pd.to_numeric(frame.get("consensus_score"), errors="coerce").fillna(0.5)
        + 0.08 * (1.0 - pd.to_numeric(frame.get("mfg_disagreement"), errors="coerce").fillna(0.5))
        + 0.08 * pd.to_numeric(frame.get("macro_alignment"), errors="coerce").fillna(0.5)
        + 0.06 * pd.to_numeric(frame.get("cone_realism"), errors="coerce").fillna(0.5)
    )
    regime_bonus = np.where(pd.to_numeric(frame.get("hmm_regime_match"), errors="coerce").fillna(0.0) > 0.5, 0.05, -0.03)
    directional_bonus = np.tanh(move * 3500.0) * 0.05
    raw = np.clip(base + regime_bonus + directional_bonus, 0.0, 1.0)
    frame["v20_cabr_score"] = raw.astype(np.float32)
    frame["v20_cabr_raw_score"] = raw.astype(np.float32)
    frame["decision_direction"] = np.where(move >= 0.0, "BUY", "SELL")
    return frame
