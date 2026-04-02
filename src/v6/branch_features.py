from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.v6.historical_retrieval import HistoricalRetrievalResult
from src.v6.regime_detection import RegimeDetectionResult
from src.v6.volatility_constraints import VolatilityEnvelope


BRANCH_FEATURE_NAMES: tuple[str, ...] = (
    "log_tft_probability",
    "regime_match",
    "volatility_match",
    "news_match",
    "crowd_match",
    "orderflow_match",
    "historical_similarity",
    "implausibility_penalty",
    "constraint_violation",
    "move_zscore",
    "vwap_deviation",
    "atr_normalized_move",
    "trend_strength",
    "momentum_persistence",
    "mean_reversion_probability",
    "breakout_probability",
    "time_since_news_score",
    "news_shock_magnitude",
    "liquidity_imbalance",
    "volume_acceleration",
    "spread_widening",
    "crowd_herding_index",
    "crowd_panic_index",
    "branch_survival_prior",
)


def _safe(mapping: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def compute_branch_feature_dict(
    branch: Mapping[str, Any],
    current_row: Mapping[str, Any],
    regime: RegimeDetectionResult,
    envelope: VolatilityEnvelope,
    retrieval: HistoricalRetrievalResult | None = None,
) -> dict[str, float]:
    current_price = max(1e-6, _safe(current_row, "close", _safe(current_row, "current_price", 1.0)))
    branch_prices = [float(value) for value in list(branch.get("predicted_prices", [])) if value is not None]
    final_price = branch_prices[-1] if branch_prices else current_price
    branch_move = final_price - current_price
    branch_move_pct = branch_move / max(current_price, 1e-6)
    atr = max(1e-6, _safe(current_row, "atr_14", current_price * max(_safe(current_row, "atr_pct", 0.001), 1e-6)))
    atr_move = branch_move / atr
    expected_abs_move = max(envelope.expected_move_abs, 1e-6)
    move_zscore = branch_move / expected_abs_move
    volatility_match = float(np.exp(-abs(abs(branch_move) - expected_abs_move) / expected_abs_move))
    constraint_violation = float(max(0.0, (abs(branch_move) - expected_abs_move) / expected_abs_move))
    implausibility_penalty = float(max(0.0, abs(move_zscore) - (2.2 + envelope.large_move_probability)))
    news_bias = _safe(current_row, "news_bias")
    crowd_bias = _safe(current_row, "crowd_bias")
    macro_bias = _safe(current_row, "macro_bias")
    llm_bias = _safe(current_row, "llm_market_bias")
    target_bias = 0.35 * macro_bias + 0.25 * news_bias + 0.20 * crowd_bias + 0.20 * llm_bias
    branch_bias = float(np.tanh(branch_move_pct * 120.0))
    regime_expectation = (
        regime.probabilities.get("trend_up", 0.0)
        - regime.probabilities.get("trend_down", 0.0)
        + 0.5 * regime.probabilities.get("breakout", 0.0) * np.sign(branch_bias or 1.0)
    )
    regime_match = float(np.exp(-abs(branch_bias - regime_expectation)))
    news_match = float(np.exp(-abs(branch_bias - target_bias)))
    crowd_match = float(np.exp(-abs(branch_bias - crowd_bias)))
    orderflow_match = float(np.exp(-abs(branch_bias - np.tanh(_safe(current_row, "displacement") + _safe(current_row, "volume_ratio") - 1.0))))
    vwap_deviation = float((_safe(current_row, "close") - _safe(current_row, "quant_kalman_fair_value", _safe(current_row, "close"))) / max(expected_abs_move, 1e-6))
    trend_strength = float(abs(_safe(current_row, "quant_trend_score")))
    momentum_persistence = float(_safe(current_row, "quant_regime_persistence"))
    mean_reversion_probability = float(regime.probabilities.get("mean_reversion", 0.0) + regime.probabilities.get("false_breakout", 0.0))
    breakout_probability = float(regime.probabilities.get("breakout", 0.0) + regime.probabilities.get("panic_news_shock", 0.0))
    time_since_news_score = float(1.0 - min(1.0, _safe(current_row, "minutes_since_news", 999.0) / 120.0))
    news_shock_magnitude = float(abs(_safe(current_row, "news_shock", _safe(current_row, "llm_event_severity", 0.0))))
    liquidity_imbalance = float(np.tanh(_safe(current_row, "dist_to_high") - _safe(current_row, "dist_to_low")))
    volume_acceleration = float(np.tanh(max(0.0, _safe(current_row, "volume_ratio") - 1.0)))
    spread_widening = float(_safe(current_row, "quant_tail_risk"))
    crowd_herding_index = float(abs(crowd_bias) * max(0.0, _safe(current_row, "crowd_extreme", 0.0)))
    crowd_panic_index = float(abs(crowd_bias) * (1.0 if crowd_bias < 0.0 else 0.4))
    branch_survival_prior = float(
        0.42 * _safe(branch, "branch_fitness")
        + 0.18 * _safe(branch, "analog_confidence")
        + 0.15 * _safe(branch, "minority_guardrail")
        + 0.10 * regime_match
        + 0.15 * volatility_match
    )
    historical_similarity = float(retrieval.similarity if retrieval is not None else _safe(branch, "analog_confidence"))
    if retrieval is not None:
        branch_survival_prior += 0.10 * retrieval.similarity + 0.08 * (1.0 - retrieval.hold_prior_30m)
        news_match = float(np.clip(news_match + 0.08 * retrieval.similarity, 0.0, 1.0))
        crowd_match = float(np.clip(crowd_match + 0.05 * abs(retrieval.directional_prior), 0.0, 1.0))
    return {
        "log_tft_probability": float(np.log(max(_safe(branch, "probability", 1e-6), 1e-6))),
        "regime_match": regime_match,
        "volatility_match": volatility_match,
        "news_match": news_match,
        "crowd_match": crowd_match,
        "orderflow_match": orderflow_match,
        "historical_similarity": historical_similarity,
        "implausibility_penalty": implausibility_penalty,
        "constraint_violation": constraint_violation,
        "move_zscore": float(move_zscore),
        "vwap_deviation": vwap_deviation,
        "atr_normalized_move": float(atr_move),
        "trend_strength": trend_strength,
        "momentum_persistence": momentum_persistence,
        "mean_reversion_probability": mean_reversion_probability,
        "breakout_probability": breakout_probability,
        "time_since_news_score": time_since_news_score,
        "news_shock_magnitude": news_shock_magnitude,
        "liquidity_imbalance": liquidity_imbalance,
        "volume_acceleration": volume_acceleration,
        "spread_widening": spread_widening,
        "crowd_herding_index": crowd_herding_index,
        "crowd_panic_index": crowd_panic_index,
        "branch_survival_prior": float(np.clip(branch_survival_prior, 0.0, 1.5)),
    }


def compute_branch_feature_vector(
    branch: Mapping[str, Any],
    current_row: Mapping[str, Any],
    regime: RegimeDetectionResult,
    envelope: VolatilityEnvelope,
    retrieval: HistoricalRetrievalResult | None = None,
) -> np.ndarray:
    feature_dict = compute_branch_feature_dict(branch, current_row, regime, envelope, retrieval)
    return np.asarray([feature_dict[name] for name in BRANCH_FEATURE_NAMES], dtype=np.float32)
