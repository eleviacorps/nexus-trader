from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REGIME_CLASSES: tuple[str, ...] = (
    "trending_up",
    "trending_down",
    "ranging",
    "breakout",
    "panic_shock",
    "low_volatility",
)


def map_regime_class(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if "bull" in normalized or "trend_up" in normalized or "trending_up" in normalized:
        return "trending_up"
    if "bear" in normalized or "trend_down" in normalized or "trending_down" in normalized:
        return "trending_down"
    if "panic" in normalized or "shock" in normalized or "news_shock" in normalized:
        return "panic_shock"
    if "breakout" in normalized:
        return "breakout"
    if "low_vol" in normalized or "quiet" in normalized:
        return "low_volatility"
    if "range" in normalized or "sideways" in normalized:
        return "ranging"
    return "ranging"


def build_regime_isolation_report(
    frame: pd.DataFrame,
    *,
    outcome_col: str = "setl_target_net_unit_pnl",
    trade_mask: np.ndarray | None = None,
    regime_col: str = "dominant_regime",
) -> dict[str, Any]:
    working = frame.copy()
    working["wfri_regime"] = working.get(regime_col, "ranging").map(map_regime_class)
    active = np.asarray(trade_mask, dtype=bool) if trade_mask is not None else np.ones(len(working), dtype=bool)
    working = working.loc[active].copy()
    aggregate = {
        "win_rate": round(float(np.mean(working[outcome_col].to_numpy(dtype=np.float32) > 0.0)) if len(working) else 0.0, 6),
        "participation": round(float(np.mean(active)) if len(active) else 0.0, 6),
    }
    by_regime: dict[str, Any] = {}
    for regime, subset in working.groupby("wfri_regime", sort=True):
        outcomes = subset[outcome_col].to_numpy(dtype=np.float32)
        trade_count = int(len(subset))
        win_rate = float(np.mean(outcomes > 0.0)) if trade_count else 0.0
        avg_unit_pnl = float(np.mean(outcomes)) if trade_count else 0.0
        by_regime[str(regime)] = {
            "win_rate": round(win_rate, 6),
            "participation": round(float(trade_count / max(len(frame), 1)), 6),
            "trade_count": trade_count,
            "avg_unit_pnl": round(avg_unit_pnl, 6),
            "deploy_recommendation": "YES" if win_rate > 0.54 and trade_count >= 50 else "NO",
        }
    return {"aggregate": aggregate, "by_regime": by_regime}
