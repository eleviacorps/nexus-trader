from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            return float(default)
        return float(number)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_text(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text or str(default)


def _direction_sign(direction: str) -> float:
    action = str(direction or "HOLD").upper()
    if action == "BUY":
        return 1.0
    if action == "SELL":
        return -1.0
    return 0.0


@dataclass(frozen=True)
class WorldState:
    timestamp: str
    symbol: str
    direction: str
    market_structure: dict[str, Any]
    nexus_features: dict[str, Any]
    quant_models: dict[str, Any]
    runtime_state: dict[str, Any]
    execution_context: dict[str, Any]

    def to_flat_features(self) -> dict[str, float]:
        numeric: dict[str, float] = {}
        for prefix, payload in (
            ("market_structure", self.market_structure),
            ("nexus_features", self.nexus_features),
            ("quant_models", self.quant_models),
            ("runtime_state", self.runtime_state),
            ("execution_context", self.execution_context),
        ):
            for key, value in payload.items():
                if isinstance(value, bool):
                    numeric[f"{prefix}_{key}"] = 1.0 if value else 0.0
                    continue
                try:
                    numeric[f"{prefix}_{key}"] = float(value)
                except Exception:
                    continue
        numeric["execution_context_direction_sign"] = _direction_sign(self.direction)
        return numeric

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_world_state(
    row: Mapping[str, Any],
    *,
    live_performance: Mapping[str, Any] | None = None,
    breaker_state: Mapping[str, Any] | None = None,
    ensemble_state: Mapping[str, Any] | None = None,
    symbol: str = "XAUUSD",
) -> WorldState:
    live = dict(live_performance or {})
    breaker = dict(breaker_state or {})
    ensemble = dict(ensemble_state or {})
    direction = _safe_text(row.get("action", row.get("decision", "HOLD")), "HOLD").upper()
    confidence_rank = _safe_int(row.get("confidence_rank"), 0)
    confidence_tier = _safe_text(row.get("confidence_tier"), "very_low").lower()
    daily_drawdown_pct = _safe_float(breaker.get("daily_drawdown_pct"), 0.0)
    if daily_drawdown_pct == 0.0:
        balance = _safe_float(live.get("balance"), 0.0)
        equity = _safe_float(live.get("equity"), balance)
        if balance > 0.0:
            daily_drawdown_pct = float((equity - balance) / balance)
    recent_direction_bias = _safe_float(live.get("recent_direction_bias"), _direction_sign(direction))
    return WorldState(
        timestamp=str(row.get("signal_time_utc", row.get("entry_time", ""))),
        symbol=str(symbol),
        direction=direction,
        market_structure={
            "close": _safe_float(row.get("reference_close", row.get("close")), 0.0),
            "atr_pct": _safe_float(row.get("atr_pct"), 0.0),
            "realized_vol_20": _safe_float(row.get("macro_realized_vol_20"), 0.0),
            "macro_vol_regime_class": _safe_int(row.get("macro_vol_regime_class"), 0),
            "return_3": _safe_float(row.get("return_3"), 0.0),
            "return_12": _safe_float(row.get("return_12"), 0.0),
            "rr_ratio": _safe_float(row.get("rr_ratio"), 0.0),
            "pnl_proxy": _safe_float(row.get("pnl_proxy"), 0.0),
        },
        nexus_features={
            "cabr_score": _safe_float(row.get("cabr_score"), 0.0),
            "cpm_score": _safe_float(row.get("cpm_score"), 0.0),
            "confidence_rank": confidence_rank,
            "confidence_tier_score": max(0.0, min(1.0, confidence_rank / 4.0)),
            "target_up_15m": _safe_float(row.get("target_up_15m"), 0.0),
        },
        quant_models={
            "hmm_confidence": _safe_float(row.get("online_hmm_regime_confidence"), 0.0),
            "hmm_persistence_count": _safe_int(row.get("online_hmm_persistence_count"), 0),
            "macro_vol_regime_class": _safe_int(row.get("macro_vol_regime_class"), 0),
            "regime_state": _safe_text(row.get("online_hmm_regime_label", row.get("hmm_state_name")), "unknown"),
        },
        runtime_state={
            "rolling_win_rate_10": _safe_float(live.get("rolling_win_rate_10"), 1.0),
            "consecutive_losses": _safe_int(live.get("consecutive_losses"), 0),
            "daily_drawdown_pct": daily_drawdown_pct,
            "daily_pnl": _safe_float(live.get("daily_pnl"), 0.0),
            "recent_direction_bias": recent_direction_bias,
            "breaker_trading_allowed": bool(breaker.get("trading_allowed", True)),
        },
        execution_context={
            "breaker_trading_allowed": bool(breaker.get("trading_allowed", True)),
            "breaker_state_paused": _safe_text(breaker.get("state"), "CLEAR").upper() == "PAUSED",
            "v22_risk_score": _safe_float(ensemble.get("risk_score"), 0.0),
            "v22_meta_label_prob": _safe_float(ensemble.get("meta_label_prob"), 0.0),
            "v22_agreement_rate": _safe_float(ensemble.get("agreement_rate"), 0.0),
            "v22_conformal_set_size": _safe_int(ensemble.get("conformal_set_size"), 0),
            "v22_max_lot": _safe_float(ensemble.get("max_lot"), 0.10),
            "rr_ratio": _safe_float(row.get("rr_ratio"), 0.0),
        },
    )


__all__ = ["WorldState", "build_world_state"]
