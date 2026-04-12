from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.v24.meta_aggregator import TradeQualityEstimate
from src.v24.world_state import WorldState


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class V24RiskJudgeConfig:
    min_expected_rr: float = 1.5
    min_profit_probability: float = 0.55
    max_uncertainty: float = 0.58
    max_danger: float = 0.62
    min_quality_score: float = -0.05
    min_hmm_confidence: float = 0.52
    wait_abstain_probability: float = 0.62
    reduce_size_danger: float = 0.48
    reduce_size_uncertainty: float = 0.42
    reduce_size_after_losses: int = 2
    default_size_multiplier: float = 0.10
    reduced_size_multiplier: float = 0.05


@dataclass(frozen=True)
class V24RiskDecision:
    action: str
    direction: str
    size_multiplier: float
    reason: str
    quality: dict[str, Any]
    runtime_flags: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class V24EnsembleRiskJudge:
    """
    Phase-1 V24 bridge: converts trade-quality outputs into execution actions.
    """

    def __init__(self, config: V24RiskJudgeConfig | None = None) -> None:
        self.config = config or V24RiskJudgeConfig()

    def decide(
        self,
        world_state: WorldState,
        quality: TradeQualityEstimate | Mapping[str, Any],
    ) -> V24RiskDecision:
        quality_map = quality.to_dict() if isinstance(quality, TradeQualityEstimate) else dict(quality)
        features = world_state.to_flat_features()
        direction = str(world_state.direction or "HOLD").upper()
        breaker_allowed = bool(features.get("execution_context_breaker_trading_allowed", 1.0) >= 0.5)
        hmm_confidence = _safe_float(features.get("quant_models_hmm_confidence"), 0.0)
        consecutive_losses = int(round(_safe_float(features.get("runtime_state_consecutive_losses"), 0.0)))
        base_size = _safe_float(features.get("execution_context_v22_max_lot"), self.config.default_size_multiplier)
        if base_size <= 0.0:
            base_size = float(self.config.default_size_multiplier)

        profit_probability = _safe_float(quality_map.get("profit_probability"), 0.0)
        expected_rr = _safe_float(quality_map.get("expected_rr"), 0.0)
        uncertainty = _safe_float(quality_map.get("uncertainty_score"), 1.0)
        danger = _safe_float(quality_map.get("danger_score"), 1.0)
        abstain_probability = _safe_float(quality_map.get("abstain_probability"), 1.0)
        quality_score = _safe_float(quality_map.get("quality_score"), -1.0)

        runtime_flags = {
            "breaker_allowed": breaker_allowed,
            "hmm_confidence": round(hmm_confidence, 6),
            "consecutive_losses": consecutive_losses,
        }
        if direction not in {"BUY", "SELL"}:
            return V24RiskDecision("WAIT", "HOLD", 0.0, "v24_no_direction", quality_map, runtime_flags)
        if not breaker_allowed:
            return V24RiskDecision("ABSTAIN", direction, 0.0, "v24_circuit_breaker", quality_map, runtime_flags)
        if expected_rr < self.config.min_expected_rr:
            return V24RiskDecision("ABSTAIN", direction, 0.0, "v24_rr_floor", quality_map, runtime_flags)
        if danger > self.config.max_danger:
            return V24RiskDecision("ABSTAIN", direction, 0.0, "v24_danger", quality_map, runtime_flags)
        if uncertainty > self.config.max_uncertainty:
            return V24RiskDecision("ABSTAIN", direction, 0.0, "v24_uncertainty", quality_map, runtime_flags)
        if profit_probability < self.config.min_profit_probability or quality_score < self.config.min_quality_score:
            return V24RiskDecision("WAIT", direction, 0.0, "v24_low_edge", quality_map, runtime_flags)
        if abstain_probability >= self.config.wait_abstain_probability or hmm_confidence < self.config.min_hmm_confidence:
            return V24RiskDecision("WAIT", direction, 0.0, "v24_regime_wait", quality_map, runtime_flags)
        if danger >= self.config.reduce_size_danger or uncertainty >= self.config.reduce_size_uncertainty or consecutive_losses >= self.config.reduce_size_after_losses:
            return V24RiskDecision(
                "REDUCE_SIZE",
                direction,
                round(min(base_size, self.config.reduced_size_multiplier), 6),
                "v24_reduce_size",
                quality_map,
                runtime_flags,
            )
        return V24RiskDecision(
            "EXECUTE",
            direction,
            round(base_size, 6),
            "v24_execute",
            quality_map,
            runtime_flags,
        )


__all__ = ["V24EnsembleRiskJudge", "V24RiskDecision", "V24RiskJudgeConfig"]
