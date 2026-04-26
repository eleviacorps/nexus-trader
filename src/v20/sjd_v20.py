from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


STANCE_LABELS = ["BUY", "SELL", "HOLD"]
CONFIDENCE_LABELS = ["LOW", "MODERATE", "HIGH"]


class SJD_V20(nn.Module):
    def __init__(self, d_features: int = 350, d_macro: int = 8, d_hidden: int = 512) -> None:
        super().__init__()
        self.macro_gate = nn.Sequential(nn.Linear(d_macro, d_features), nn.Sigmoid())
        self.encoder = nn.Sequential(
            nn.Linear(d_features, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 256),
            nn.GELU(),
        )
        self.stance_head = nn.Linear(256, 3)
        self.confidence_head = nn.Linear(256, 3)
        self.tp_head = nn.Linear(256, 1)
        self.sl_head = nn.Linear(256, 1)
        self.kelly_head = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor, macro_context: torch.Tensor) -> dict[str, torch.Tensor]:
        gate = self.macro_gate(macro_context)
        gated_features = features * gate
        h = self.encoder(gated_features)
        return {
            "stance": self.stance_head(h),
            "confidence": self.confidence_head(h),
            "tp_offset": self.tp_head(h).squeeze(-1),
            "sl_offset": self.sl_head(h).squeeze(-1),
            "kelly": torch.sigmoid(self.kelly_head(h)).squeeze(-1) * 0.25,
        }


def rule_based_sjd_labels(feature_frame: pd.DataFrame) -> pd.DataFrame:
    frame = feature_frame.copy()
    future_return = pd.to_numeric(frame.get("future_return_15m"), errors="coerce").fillna(0.0)
    hmm_state = pd.to_numeric(frame.get("hmm_state"), errors="coerce").fillna(0).astype(int)
    hurst = pd.to_numeric(frame.get("hurst_overall"), errors="coerce").fillna(0.5)
    disagreement = pd.to_numeric(frame.get("mfg_disagreement"), errors="coerce").fillna(0.5)
    stance = np.where(
        ((hmm_state == 5) & (disagreement > 0.7)),
        "HOLD",
        np.where(future_return > 0.0, "BUY", np.where(future_return < 0.0, "SELL", "HOLD")),
    )
    confidence = np.where(
        ((hmm_state == 1) | (hmm_state == 3)) & (hurst > 0.55) & (disagreement < 0.3),
        "HIGH",
        np.where((hmm_state == 5) & (disagreement > 0.7), "LOW", "MODERATE"),
    )
    tp_offset = np.clip(pd.to_numeric(frame.get("atr_pct"), errors="coerce").fillna(0.001) * 1200.0, 15.0, 90.0)
    sl_offset = np.clip(tp_offset * 0.66, 10.0, 60.0)
    kelly = np.clip(
        0.02
        + 0.10 * (confidence == "HIGH").astype(float)
        + 0.05 * (confidence == "MODERATE").astype(float)
        - 0.04 * (disagreement > 0.6).astype(float),
        0.0,
        0.25,
    )
    return pd.DataFrame(
        {
            "stance": stance,
            "confidence": confidence,
            "tp_offset": tp_offset.astype(np.float32),
            "sl_offset": sl_offset.astype(np.float32),
            "kelly": kelly.astype(np.float32),
        },
        index=frame.index,
    )


@dataclass
class SJDDecision:
    final_call: str
    confidence: str
    tp_offset: float
    sl_offset: float
    kelly: float
    reasoning: str


def latest_sjd_decision(feature_row: dict[str, Any]) -> SJDDecision:
    direction_signal = float(feature_row.get("direction_signal", 0.0))
    hmm_state = int(float(feature_row.get("hmm_state", 0)))
    hurst = float(feature_row.get("hurst_overall", 0.5))
    disagreement = float(feature_row.get("mfg_disagreement", 0.5))
    confidence = "HIGH" if hmm_state in {1, 3} and hurst > 0.55 and disagreement < 0.30 else "LOW" if hmm_state == 5 and disagreement > 0.70 else "MODERATE"
    final_call = "BUY" if direction_signal > 0.08 else "SELL" if direction_signal < -0.08 else "SKIP"
    tp_offset = float(np.clip(abs(float(feature_row.get("atr_pct", 0.001))) * 1200.0, 15.0, 90.0))
    sl_offset = float(np.clip(tp_offset * 0.66, 10.0, 60.0))
    kelly = float(np.clip(0.03 + (0.08 if confidence == "HIGH" else 0.03 if confidence == "MODERATE" else 0.0), 0.0, 0.25))
    reasoning = f"HMM state {hmm_state}, Hurst {hurst:.3f}, and MFG disagreement {disagreement:.3f} imply {confidence.lower()} conviction."
    return SJDDecision(final_call=final_call, confidence=confidence, tp_offset=tp_offset, sl_offset=sl_offset, kelly=kelly, reasoning=reasoning)
