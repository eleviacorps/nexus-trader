from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def deflated_sharpe(returns: np.ndarray, n_trials: int = 1) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    if std <= 0.0:
        return 0.0
    sharpe = mean / std
    penalty = np.sqrt(max(np.log(max(n_trials, 1)), 0.0) / max(returns.size, 1))
    return float(sharpe - penalty)


def summarize_trade_frame(trades: pd.DataFrame, *, n_trials: int = 1) -> dict[str, Any]:
    pnl = pd.to_numeric(trades.get("pnl_usd"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    returns = pd.to_numeric(trades.get("return_pct"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    gross_positive = float(pnl[pnl > 0.0].sum()) if np.any(pnl > 0.0) else 0.0
    gross_negative = float(abs(pnl[pnl < 0.0].sum())) if np.any(pnl < 0.0) else 0.0
    equity_curve = 1000.0 + np.cumsum(pnl)
    peaks = np.maximum.accumulate(equity_curve) if equity_curve.size else np.asarray([1000.0])
    drawdown = np.max((peaks - equity_curve) / peaks) if equity_curve.size else 0.0
    return {
        "annual_sharpe_ratio": round(deflated_sharpe(returns, n_trials=n_trials), 6),
        "calmar_ratio": round(float(np.mean(returns) / max(drawdown, 1e-6)) if returns.size else 0.0, 6),
        "annual_sortino_ratio": round(float(np.mean(returns) / max(np.std(returns[returns < 0.0], ddof=0), 1e-6)) if np.any(returns < 0.0) else 0.0, 6),
        "max_drawdown": round(float(drawdown), 6),
        "profit_factor": round(float(gross_positive / gross_negative) if gross_negative > 0.0 else 0.0, 6),
        "win_rate": round(float(np.mean(pnl > 0.0)) if pnl.size else 0.0, 6),
    }


def save_walkforward(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
