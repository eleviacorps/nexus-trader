from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from src.v14.bst import branch_survival_score
from src.v15.cpm import ConditionalPredictabilityMapper
from src.v16.confidence_tier import ConfidenceTier, TIER_COLORS, TIER_LABELS, classify_confidence
from src.v16.sel import sel_lot_size, should_execute
from src.v16.sqt import SimulationQualityTracker


SYMBOL_PIP_SIZE = {
    "XAUUSD": 0.1,
    "EURUSD": 0.0001,
    "BTCUSD": 1.0,
}


@dataclass(frozen=True)
class SimulationResult:
    timestamp: str
    symbol: str
    direction: str
    confidence_tier: ConfidenceTier
    tier_color: str
    tier_label: str
    consensus_path: list[float]
    cone_upper: list[float]
    cone_lower: list[float]
    minority_path: list[float]
    minority_direction: str
    selected_branch_idx: int
    cabr_score: float
    bst_score: float
    cone_width_pips: float
    cpm_score: float
    crowd_persona: str
    n_branches: int
    sqt_label: str
    sqt_accuracy: float
    mode: str
    primary_horizon_minutes: int
    refresh_interval_minutes: int
    should_execute: bool
    execution_reason: str
    suggested_lot: float

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence_tier"] = self.confidence_tier.value
        return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _symbol_pip_size(symbol: str) -> float:
    return float(SYMBOL_PIP_SIZE.get(str(symbol).upper(), 0.1))


def _direction_from_prices(anchor_price: float, target_price: float) -> str:
    if float(target_price) > float(anchor_price):
        return "BUY"
    if float(target_price) < float(anchor_price):
        return "SELL"
    return "NEUTRAL"


def _horizon_cone(cone: list[Mapping[str, Any]], max_minutes: int) -> list[Mapping[str, Any]]:
    filtered = []
    for point in cone:
        minutes = int(round(_safe_float(point.get("horizon"), 0.0) * 5.0))
        if minutes <= 0 or minutes > int(max_minutes):
            continue
        filtered.append(point)
    return filtered


def _minority_path(branches: list[Mapping[str, Any]], highlighted: Mapping[str, Any], anchor_price: float, max_minutes: int) -> tuple[list[float], str]:
    minority_id = highlighted.get("minority_branch_id")
    for branch in branches:
        if branch.get("path_id") != minority_id:
            continue
        prices = [float(anchor_price)]
        for index, price in enumerate(branch.get("predicted_prices", []), start=1):
            if index * 5 > int(max_minutes):
                break
            prices.append(float(price))
        direction = _direction_from_prices(prices[0], prices[-1]) if prices else "NEUTRAL"
        return prices, direction
    return [float(anchor_price)], "NEUTRAL"


def _cabr_proxy(branches: list[Mapping[str, Any]], simulation: Mapping[str, Any], model_prediction: Mapping[str, Any] | None) -> float:
    top_branch = branches[0] if branches else {}
    selector_score = _safe_float(top_branch.get("selector_score"), _safe_float(simulation.get("selector_top_score"), 0.0))
    probability = _safe_float(top_branch.get("probability"), _safe_float(simulation.get("mean_probability"), 0.5))
    model_15 = _safe_float(((model_prediction or {}).get("horizon_probabilities") or {}).get("15m"), _safe_float((model_prediction or {}).get("bullish_probability"), 0.5))
    confidence = _safe_float(simulation.get("overall_confidence"), 0.0)
    blended = (0.34 * probability) + (0.28 * selector_score) + (0.20 * confidence) + (0.18 * max(model_15, 1.0 - model_15))
    return float(np.clip(0.45 + (0.50 * blended), 0.50, 0.95))


def build_v16_simulation_result(
    payload: Mapping[str, Any],
    model_prediction: Mapping[str, Any] | None,
    *,
    mode: str = "frequency",
    sqt: SimulationQualityTracker | None = None,
    eci_context: Mapping[str, Any] | None = None,
    primary_horizon_minutes: int = 15,
) -> dict[str, Any]:
    market = payload.get("market", {}) if isinstance(payload, Mapping) else {}
    simulation = payload.get("simulation", {}) if isinstance(payload, Mapping) else {}
    current_row = payload.get("current_row", {}) if isinstance(payload, Mapping) else {}
    symbol = str(payload.get("symbol", "XAUUSD"))
    anchor_price = _safe_float(market.get("current_price"), 0.0)
    cone = _horizon_cone(list(payload.get("cone", [])), max_minutes=primary_horizon_minutes)
    if not cone:
        cone = [
            {
                "timestamp": market.get("candles", [{}])[-1].get("timestamp"),
                "center_price": anchor_price,
                "lower_price": anchor_price,
                "upper_price": anchor_price,
            }
        ]
    modifier = _safe_float((eci_context or {}).get("cone_width_modifier"), 0.0)
    consensus_path = [round(anchor_price, 5)] + [round(_safe_float(point.get("center_price"), anchor_price), 5) for point in cone]
    cone_upper = [round(anchor_price, 5)]
    cone_lower = [round(anchor_price, 5)]
    for point in cone:
        center = _safe_float(point.get("center_price"), anchor_price)
        lower = _safe_float(point.get("lower_price"), center)
        upper = _safe_float(point.get("upper_price"), center)
        half_width = max(upper - center, center - lower)
        widened_half_width = half_width * max(1.0, 1.0 + modifier)
        cone_upper.append(round(center + widened_half_width, 5))
        cone_lower.append(round(max(0.0, center - widened_half_width), 5))

    minority_path, minority_direction = _minority_path(
        list(payload.get("branches", [])),
        payload.get("highlighted_branches", {}) if isinstance(payload, Mapping) else {},
        anchor_price,
        primary_horizon_minutes,
    )
    branches = list(payload.get("branches", [])) if isinstance(payload, Mapping) else []
    top_branch = branches[0] if branches else {}
    top_branch_prices = np.asarray([anchor_price] + [float(item) for item in top_branch.get("predicted_prices", [])[: max(1, primary_horizon_minutes // 5)]], dtype=np.float32)
    atr = max(_safe_float(current_row.get("atr_14"), anchor_price * 0.0015), 0.25)
    bst_score = float(branch_survival_score(top_branch_prices, current_atr=atr, n_perturbations=30)) if top_branch_prices.size >= 2 else 0.5
    cabr_score = _cabr_proxy(branches, simulation, model_prediction)
    cpm_score = float(ConditionalPredictabilityMapper().score_row(current_row).get("predictability", 0.5))
    pip_size = _symbol_pip_size(symbol)
    cone_width_pips = 0.0
    if len(cone_upper) > 1 and len(cone_lower) > 1:
        cone_width_pips = float(np.mean(np.asarray(cone_upper[1:], dtype=np.float64) - np.asarray(cone_lower[1:], dtype=np.float64)) / max(pip_size, 1e-9))
    tier = classify_confidence(cabr_score, bst_score, cone_width_pips, cpm_score)
    tracker = sqt or SimulationQualityTracker()
    execute, execution_reason = should_execute(tier, mode, tracker)
    suggested_lot = sel_lot_size(
        equity=max(_safe_float(payload.get("paper_trading", {}).get("equity"), _safe_float(payload.get("paper_trading", {}).get("balance"), 1000.0)), 100.0),
        confidence_tier=tier,
        sqt_label=str(tracker.summary().get("label", "NEUTRAL")),
        mode=mode,
    )
    result = SimulationResult(
        timestamp=str(payload.get("generated_at", "")),
        symbol=symbol,
        direction=_direction_from_prices(consensus_path[0], consensus_path[-1]),
        confidence_tier=tier,
        tier_color=TIER_COLORS[tier],
        tier_label=TIER_LABELS[tier],
        consensus_path=consensus_path,
        cone_upper=cone_upper,
        cone_lower=cone_lower,
        minority_path=[round(float(value), 5) for value in minority_path],
        minority_direction=minority_direction,
        selected_branch_idx=int(_safe_float(top_branch.get("path_id"), 1)) - 1 if top_branch else 0,
        cabr_score=round(cabr_score, 6),
        bst_score=round(bst_score, 6),
        cone_width_pips=round(cone_width_pips, 3),
        cpm_score=round(cpm_score, 6),
        crowd_persona=str(top_branch.get("dominant_persona", "mixed")),
        n_branches=len(branches),
        sqt_label=str(tracker.summary().get("label", "NEUTRAL")),
        sqt_accuracy=float(tracker.summary().get("rolling_accuracy", 0.5) or 0.5),
        mode=str(mode).strip().lower() or "frequency",
        primary_horizon_minutes=int(primary_horizon_minutes),
        refresh_interval_minutes=5,
        should_execute=bool(execute),
        execution_reason=execution_reason,
        suggested_lot=float(suggested_lot),
    )
    result_payload = result.to_payload()
    return {
        "simulation": result_payload,
        "final_forecast": {
            "mode": "v16_simulator",
            "points": [
                {
                    "minutes": index * 5,
                    "timestamp": cone[index - 1].get("timestamp") if index - 1 < len(cone) else None,
                    "final_price": consensus_path[index],
                    "cone_upper": cone_upper[index],
                    "cone_lower": cone_lower[index],
                    "minority_price": result_payload["minority_path"][index] if index < len(result_payload["minority_path"]) else result_payload["minority_path"][-1],
                }
                for index in range(1, len(consensus_path))
            ],
            "horizon_table": [
                {
                    "minutes": index * 5,
                    "final_price": consensus_path[index],
                    "cone_upper": cone_upper[index],
                    "cone_lower": cone_lower[index],
                    "minority_price": result_payload["minority_path"][index] if index < len(result_payload["minority_path"]) else result_payload["minority_path"][-1],
                }
                for index in range(1, len(consensus_path))
                if index * 5 in {5, 10, 15}
            ],
        },
    }
