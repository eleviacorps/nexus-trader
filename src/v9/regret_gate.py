from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegretGateResult:
    should_trade: np.ndarray
    regret_margin: np.ndarray
    dynamic_threshold: np.ndarray


def regret_gate(
    pred_confidence: np.ndarray | float,
    projected_move_pips: np.ndarray | float,
    *,
    spread_pips: float = 0.5,
) -> RegretGateResult:
    confidence = np.clip(np.asarray(pred_confidence, dtype=np.float32), 0.0, 1.0)
    projected_move = np.maximum(np.asarray(projected_move_pips, dtype=np.float32), 1e-3)
    cost_of_wrong = projected_move + float(spread_pips)
    cost_of_missing = projected_move * confidence
    expected_regret_trade = cost_of_wrong * (1.0 - confidence)
    expected_regret_abstain = cost_of_missing
    dynamic_threshold = (projected_move + float(spread_pips)) / np.maximum((2.0 * projected_move) + float(spread_pips), 1e-3)
    should_trade = confidence >= dynamic_threshold
    regret_margin = expected_regret_abstain - expected_regret_trade
    return RegretGateResult(
        should_trade=np.asarray(should_trade, dtype=bool),
        regret_margin=np.asarray(regret_margin, dtype=np.float32),
        dynamic_threshold=np.asarray(dynamic_threshold, dtype=np.float32),
    )


def projected_move_from_context(
    probabilities: np.ndarray,
    *,
    confidence_probabilities: np.ndarray | None = None,
    context_features: np.ndarray | None = None,
) -> np.ndarray:
    direction_conf = np.abs(np.asarray(probabilities, dtype=np.float32) - 0.5) * 2.0
    if confidence_probabilities is not None:
        direction_conf = 0.55 * direction_conf + 0.45 * np.clip(np.asarray(confidence_probabilities, dtype=np.float32), 0.0, 1.0)
    context_scale = np.ones_like(direction_conf, dtype=np.float32)
    if context_features is not None and context_features.ndim == 2 and context_features.shape[0] == direction_conf.shape[0]:
        atr_proxy = np.clip(context_features[:, 0], 0.0, 1.5)
        vol_proxy = np.clip(context_features[:, 9], 0.0, 1.0)
        context_scale = 6.0 + 10.0 * atr_proxy + 6.0 * vol_proxy
    else:
        context_scale = np.full_like(direction_conf, 10.0, dtype=np.float32)
    return np.maximum(1.0, context_scale * (0.35 + 0.65 * direction_conf)).astype(np.float32)


def regret_gate_scores(
    probabilities: np.ndarray,
    *,
    confidence_probabilities: np.ndarray | None = None,
    context_features: np.ndarray | None = None,
    spread_pips: float = 0.5,
) -> np.ndarray:
    projected_move = projected_move_from_context(
        probabilities,
        confidence_probabilities=confidence_probabilities,
        context_features=context_features,
    )
    confidence = np.abs(np.asarray(probabilities, dtype=np.float32) - 0.5) * 2.0
    if confidence_probabilities is not None:
        confidence = 0.55 * confidence + 0.45 * np.clip(np.asarray(confidence_probabilities, dtype=np.float32), 0.0, 1.0)
    result = regret_gate(confidence, projected_move, spread_pips=spread_pips)
    score = 1.0 / (1.0 + np.exp(-np.clip(result.regret_margin, -8.0, 8.0)))
    return np.asarray(score, dtype=np.float32)
