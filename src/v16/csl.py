from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from src.v14.bst import branch_survival_score
from src.v15.cpm import ConditionalPredictabilityMapper
from src.v16.confidence_tier import ConfidenceTier, TIER_COLORS, TIER_LABELS, classify_confidence
from src.v16.sel import sel_lot_size, should_execute
from src.v16.sqt import SimulationQualityTracker
from src.v17.relativistic_cone import RelativisticCone


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
    cone_outer_upper: list[float]
    cone_outer_lower: list[float]
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
    cone_type: str
    cone_c_m: float
    cone_h_plus: float
    cone_h_minus: float
    cone_compact_support: bool
    cone_asymmetric: bool

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


def _historical_log_returns(candles: list[Mapping[str, Any]]) -> np.ndarray:
    closes = np.asarray([_safe_float(item.get("close"), 0.0) for item in candles], dtype=np.float64)
    closes = closes[closes > 0.0]
    if closes.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(np.log(closes))


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
    raw_cone = payload.get("cone", [])
    if isinstance(raw_cone, Mapping):
        raw_cone = list(raw_cone.get("points", []))
    cone = _horizon_cone(list(raw_cone), max_minutes=primary_horizon_minutes)
    if not cone:
        cone = [
            {
                "timestamp": market.get("candles", [{}])[-1].get("timestamp"),
                "center_price": anchor_price,
                "lower_price": anchor_price,
                "upper_price": anchor_price,
            }
        ]
    consensus_path = [round(anchor_price, 5)] + [round(_safe_float(point.get("center_price"), anchor_price), 5) for point in cone]
    hurst_positive = _safe_float(current_row.get("hurst_positive"), 0.55)
    hurst_negative = _safe_float(current_row.get("hurst_negative"), 0.62)
    historical_returns = _historical_log_returns(list(market.get("candles", [])))
    rc = RelativisticCone(historical_returns)
    rc_envelope = rc.envelope(
        current_price=anchor_price,
        n_bars=max(len(cone), 1),
        hurst_positive=hurst_positive,
        hurst_negative=hurst_negative,
    )
    inner_upper = [round(anchor_price, 5)] + [round(float(value), 5) for value in list(rc_envelope.get("inner_upper", []))]
    inner_lower = [round(anchor_price, 5)] + [round(float(value), 5) for value in list(rc_envelope.get("inner_lower", []))]
    outer_upper = [round(anchor_price, 5)] + [round(float(value), 5) for value in list(rc_envelope.get("cone_upper", []))]
    outer_lower = [round(anchor_price, 5)] + [round(float(value), 5) for value in list(rc_envelope.get("cone_lower", []))]
    cone_points = []
    for index, point in enumerate(cone, start=1):
        cone_points.append(
            {
                "timestamp": point.get("timestamp"),
                "horizon": index,
                "center_price": consensus_path[index],
                "lower_price": inner_lower[index] if index < len(inner_lower) else inner_lower[-1],
                "upper_price": inner_upper[index] if index < len(inner_upper) else inner_upper[-1],
                "outer_lower_price": outer_lower[index] if index < len(outer_lower) else outer_lower[-1],
                "outer_upper_price": outer_upper[index] if index < len(outer_upper) else outer_upper[-1],
            }
        )

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
    legacy_bst = float(branch_survival_score(top_branch_prices, current_atr=atr, n_perturbations=30)) if top_branch_prices.size >= 2 else 0.5
    branch_log_returns = np.diff(np.log(np.clip(top_branch_prices.astype(np.float64), 1e-8, None))) if top_branch_prices.size >= 2 else np.asarray([], dtype=np.float64)
    branch_plausibility = rc.branch_plausibility(branch_log_returns, n_bars=max(1, primary_horizon_minutes // 5))
    bst_score = float(np.clip((0.55 * legacy_bst) + (0.45 * branch_plausibility), 0.0, 1.0))
    cabr_score = _cabr_proxy(branches, simulation, model_prediction)
    cpm_score = float(ConditionalPredictabilityMapper().score_row(current_row).get("predictability", 0.5))
    pip_size = _symbol_pip_size(symbol)
    cone_width_pips = 0.0
    if len(inner_upper) > 1 and len(inner_lower) > 1:
        cone_width_pips = float(np.mean(np.asarray(inner_upper[1:], dtype=np.float64) - np.asarray(inner_lower[1:], dtype=np.float64)) / max(pip_size, 1e-9))
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
        cone_upper=inner_upper,
        cone_lower=inner_lower,
        cone_outer_upper=outer_upper,
        cone_outer_lower=outer_lower,
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
        cone_type="relativistic",
        cone_c_m=float(rc_envelope.get("c_m", 0.003)),
        cone_h_plus=float(rc_envelope.get("h_plus", hurst_positive)),
        cone_h_minus=float(rc_envelope.get("h_minus", hurst_negative)),
        cone_compact_support=bool(rc_envelope.get("compact_support", True)),
        cone_asymmetric=bool(rc_envelope.get("asymmetric", False)),
    )
    result_payload = result.to_payload()
    return {
        "simulation": result_payload,
        "cone": {
            "type": "relativistic",
            "points": cone_points,
            "outer_upper": outer_upper[1:],
            "outer_lower": outer_lower[1:],
            "inner_upper": inner_upper[1:],
            "inner_lower": inner_lower[1:],
            "c_m": rc_envelope.get("c_m", 0.003),
            "h_plus": rc_envelope.get("h_plus", hurst_positive),
            "h_minus": rc_envelope.get("h_minus", hurst_negative),
            "compact_support": rc_envelope.get("compact_support", True),
            "asymmetric": rc_envelope.get("asymmetric", False),
        },
        "relativistic_cone": {
            "type": "relativistic",
            "points": cone_points,
            "outer_upper": outer_upper[1:],
            "outer_lower": outer_lower[1:],
            "inner_upper": inner_upper[1:],
            "inner_lower": inner_lower[1:],
            "c_m": rc_envelope.get("c_m", 0.003),
            "h_plus": rc_envelope.get("h_plus", hurst_positive),
            "h_minus": rc_envelope.get("h_minus", hurst_negative),
            "compact_support": rc_envelope.get("compact_support", True),
            "asymmetric": rc_envelope.get("asymmetric", False),
        },
        "final_forecast": {
            "mode": "v16_simulator",
            "points": [
                {
                    "minutes": index * 5,
                    "timestamp": cone_points[index - 1].get("timestamp") if index - 1 < len(cone_points) else None,
                    "final_price": consensus_path[index],
                    "cone_upper": inner_upper[index],
                    "cone_lower": inner_lower[index],
                    "outer_upper": outer_upper[index],
                    "outer_lower": outer_lower[index],
                    "minority_price": result_payload["minority_path"][index] if index < len(result_payload["minority_path"]) else result_payload["minority_path"][-1],
                }
                for index in range(1, len(consensus_path))
            ],
            "horizon_table": [
                {
                    "minutes": index * 5,
                    "final_price": consensus_path[index],
                    "cone_upper": inner_upper[index],
                    "cone_lower": inner_lower[index],
                    "outer_upper": outer_upper[index],
                    "outer_lower": outer_lower[index],
                    "minority_price": result_payload["minority_path"][index] if index < len(result_payload["minority_path"]) else result_payload["minority_path"][-1],
                }
                for index in range(1, len(consensus_path))
                if index * 5 in {5, 10, 15}
            ],
        },
    }
