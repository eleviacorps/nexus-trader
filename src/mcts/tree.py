from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping

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
    consensus = float(current_row.get('consensus_score', 0.0))
    consensus_score = 0.5 + 0.5 * min(1.0, abs(consensus))
    return max(
        0.0,
        min(
            1.0,
            0.28 * vol_score
            + 0.24 * body_score
            + 0.16 * wick_balance
            + 0.16 * directional_alignment
            + 0.10 * macro_alignment
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
    root = SimulationNode(seed=root_seed, depth=0, probability_weight=1.0, dominant_driver='root')

    def _expand(node: SimulationNode) -> None:
        if node.depth >= max_depth:
            return
        for branch in range(2):
            seed = node.seed * 10 + branch + 1
            state = simulate_one_step(current_row=current_row, personas=personas, seed=seed)
            score = score_state(state, current_row)
            child = SimulationNode(
                seed=seed,
                depth=node.depth + 1,
                probability_weight=node.probability_weight * 0.5 * (0.5 + score / 2),
                dominant_driver=_dominant_driver(state),
                state=state,
                path_prices=[*node.path_prices, state.close],
            )
            node.children.append(child)
            _expand(child)

    _expand(root)
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
