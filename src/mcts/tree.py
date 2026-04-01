from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping

from src.simulation.abm import SyntheticMarketState, simulate_one_step
from src.simulation.personas import Persona


@dataclass
class SimulationNode:
    seed: int
    depth: int
    probability_weight: float
    dominant_driver: str
    state: SyntheticMarketState | None = None
    path_prices: List[float] = field(default_factory=list)
    children: List['SimulationNode'] = field(default_factory=list)
    branch_fitness: float = 0.0
    branch_label: str = "balanced"
    minority_guardrail: float = 0.0
    row_snapshot: dict[str, float] = field(default_factory=dict)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _branch_label(directional_bias: float, current_row: Mapping[str, float]) -> str:
    macro_bias = float(current_row.get("macro_bias", 0.0) or 0.0)
    crowd_bias = float(current_row.get("crowd_bias", 0.0) or 0.0)
    llm_bias = float(current_row.get("llm_market_bias", 0.0) or 0.0)
    if directional_bias >= 0.18:
        if crowd_bias < -0.2:
            return "reversal_bid"
        if macro_bias > 0.15 or llm_bias > 0.15:
            return "macro_markup"
        return "trend_continuation_up"
    if directional_bias <= -0.18:
        if crowd_bias > 0.2:
            return "reversal_flush"
        if macro_bias < -0.15 or llm_bias < -0.15:
            return "macro_breakdown"
        return "trend_continuation_down"
    return "balanced_probe"


def _roll_forward_row(current_row: Mapping[str, float], state: SyntheticMarketState) -> dict[str, float]:
    next_row = {key: float(value) for key, value in current_row.items() if isinstance(value, (int, float))}
    prev_close = float(current_row.get("close", state.open) or state.open)
    close = float(state.close)
    high = float(state.high)
    low = float(state.low)
    open_ = float(state.open)
    candle_range = max(high - low, 1e-6)
    body = close - open_
    atr = max(float(current_row.get("atr_14", candle_range) or candle_range), 1e-6)
    new_atr = max(1e-6, 0.82 * atr + 0.18 * candle_range)
    displacement = body / new_atr
    return_1 = 0.0 if abs(prev_close) <= 1e-9 else (close - prev_close) / prev_close
    ema_cross = _clamp(0.85 * float(current_row.get("ema_cross", 0.0) or 0.0) + 1.25 * displacement, -1.0, 1.0)
    macd_hist = 0.75 * float(current_row.get("macd_hist", 0.0) or 0.0) + 0.9 * displacement
    macd = 0.8 * float(current_row.get("macd", 0.0) or 0.0) + 0.65 * displacement
    macd_sig = 0.85 * float(current_row.get("macd_sig", 0.0) or 0.0) + 0.35 * macd
    rsi_14 = _clamp(float(current_row.get("rsi_14", 50.0) or 50.0) + displacement * 18.0, 1.0, 99.0)
    rsi_7 = _clamp(float(current_row.get("rsi_7", 50.0) or 50.0) + displacement * 24.0, 1.0, 99.0)
    bb_pct = _clamp(float(current_row.get("bb_pct", 0.5) or 0.5) + displacement * 0.16, 0.0, 1.0)
    dist_to_high = max(0.0, 0.68 * float(current_row.get("dist_to_high", 1.5) or 1.5) - max(displacement, 0.0) * 0.9)
    dist_to_low = max(0.0, 0.68 * float(current_row.get("dist_to_low", 1.5) or 1.5) + min(displacement, 0.0) * 0.9)
    upper_wick = (high - max(open_, close)) / candle_range
    lower_wick = (min(open_, close) - low) / candle_range
    body_pct = min(1.0, abs(body) / candle_range)

    next_row.update(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr_14": new_atr,
            "atr_pct": new_atr / max(abs(close), 1e-6),
            "return_1": return_1,
            "return_3": 0.7 * float(current_row.get("return_1", 0.0) or 0.0) + 0.3 * return_1,
            "return_6": 0.8 * float(current_row.get("return_3", 0.0) or 0.0) + 0.2 * return_1,
            "return_12": 0.85 * float(current_row.get("return_6", 0.0) or 0.0) + 0.15 * return_1,
            "ema_cross": ema_cross,
            "macd_hist": macd_hist,
            "macd": macd,
            "macd_sig": macd_sig,
            "rsi_14": rsi_14,
            "rsi_7": rsi_7,
            "bb_pct": bb_pct,
            "body_pct": body_pct,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "displacement": displacement,
            "dist_to_high": dist_to_high,
            "dist_to_low": dist_to_low,
            "hh": 1.0 if dist_to_high <= 0.12 and close >= prev_close else 0.0,
            "ll": 1.0 if dist_to_low <= 0.12 and close <= prev_close else 0.0,
            "is_bullish": 1.0 if close >= open_ else 0.0,
            "consensus_score": abs(float(state.directional_bias)),
        }
    )
    return next_row


def score_state(state: SyntheticMarketState, current_row: Mapping[str, float]) -> float:
    atr = max(float(current_row.get('atr_14', 0.0) or 0.0), 1e-6)
    if atr <= 1e-6:
        atr_pct = float(current_row.get('atr_pct', 0.0) or 0.0)
        current_price = max(float(current_row.get('close', 1.0) or 1.0), 1e-6)
        atr = max(current_price * max(atr_pct, 1e-6), 1e-6)
    candle_range = state.high - state.low
    body = abs(state.close - state.open)
    upper_wick = state.high - max(state.open, state.close)
    lower_wick = min(state.open, state.close) - state.low
    range_ratio = candle_range / atr
    vol_score = 1.0 - min(1.0, abs(range_ratio - 1.0) / 2.0)
    body_score = min(1.0, (body / candle_range) * 1.5) if candle_range > 0 else 0.0
    wick_balance = 1.0 - abs(upper_wick - lower_wick) / candle_range if candle_range > 0 else 0.5
    directional_alignment = 0.5 + 0.5 * abs(float(state.directional_bias))
    macro_bias = float(current_row.get('macro_bias', 0.0))
    macro_alignment = 1.0 - min(1.0, abs(macro_bias - float(state.directional_bias)) / 2.0)
    llm_bias = float(current_row.get("llm_market_bias", 0.0) or 0.0)
    news_bias = float(current_row.get("news_bias", 0.0) or 0.0)
    crowd_bias = float(current_row.get("crowd_bias", 0.0) or 0.0)
    crowd_extreme = float(current_row.get("crowd_extreme", 0.0) or 0.0)
    target_bias = (0.45 * macro_bias) + (0.35 * llm_bias) + (0.20 * news_bias)
    narrative_alignment = 1.0 - min(1.0, abs(target_bias - float(state.directional_bias)) / 2.0)
    crowd_fade_bonus = 1.0 - min(1.0, abs((crowd_bias * crowd_extreme) + float(state.directional_bias)) / 2.0)
    consensus = float(current_row.get('consensus_score', 0.0))
    consensus_score = 0.5 + 0.5 * min(1.0, abs(consensus))
    return max(
        0.0,
        min(
            1.0,
            0.24 * vol_score
            + 0.18 * body_score
            + 0.12 * wick_balance
            + 0.16 * directional_alignment
            + 0.11 * macro_alignment
            + 0.11 * narrative_alignment
            + 0.08 * crowd_fade_bonus
            + 0.06 * consensus_score,
        ),
    )


def _dominant_driver(state: SyntheticMarketState) -> str:
    if state.buy_pressure == state.sell_pressure:
        return 'balanced'
    return 'crowd_buying' if state.buy_pressure > state.sell_pressure else 'crowd_selling'


def dominant_persona_name(state: SyntheticMarketState) -> str:
    if not state.decisions:
        return 'unknown'
    best = max(state.decisions, key=lambda decision: abs(decision.impact))
    return best.persona


def expand_binary_tree(
    current_row: Mapping[str, float],
    personas: Mapping[str, Persona],
    max_depth: int = 5,
    root_seed: int = 42,
) -> SimulationNode:
    root = SimulationNode(
        seed=root_seed,
        depth=0,
        probability_weight=1.0,
        dominant_driver='root',
        row_snapshot={key: float(value) for key, value in current_row.items() if isinstance(value, (int, float))},
    )

    def _expand(node: SimulationNode, row_context: Mapping[str, float]) -> None:
        if node.depth >= max_depth:
            return
        for branch in range(2):
            seed = node.seed * 10 + branch + 1
            state = simulate_one_step(current_row=row_context, personas=personas, seed=seed)
            score = score_state(state, row_context)
            next_row = _roll_forward_row(row_context, state)
            minority_guardrail = abs(float(next_row.get("crowd_bias", 0.0) or 0.0)) * (
                1.0 if float(state.directional_bias) * float(next_row.get("crowd_bias", 0.0) or 0.0) < 0.0 else 0.35
            )
            child = SimulationNode(
                seed=seed,
                depth=node.depth + 1,
                probability_weight=node.probability_weight * 0.5 * (0.45 + score / 1.8 + minority_guardrail * 0.08),
                dominant_driver=_dominant_driver(state),
                state=state,
                path_prices=[*node.path_prices, state.close],
                branch_fitness=score,
                branch_label=_branch_label(float(state.directional_bias), row_context),
                minority_guardrail=_clamp(minority_guardrail, 0.0, 1.0),
                row_snapshot=next_row,
            )
            node.children.append(child)
            _expand(child, next_row)

    _expand(root, current_row)
    return root


def iter_leaves(node: SimulationNode) -> List[SimulationNode]:
    if not node.children:
        return [node]
    leaves: List[SimulationNode] = []
    for child in node.children:
        leaves.extend(iter_leaves(child))
    return leaves


def assert_leaf_count(node: SimulationNode, expected: int = 32) -> None:
    leaves = iter_leaves(node)
    if len(leaves) != expected:
        raise AssertionError(f'Expected {expected} leaves, found {len(leaves)}')
