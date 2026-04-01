from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Sequence

from src.mcts.tree import SimulationNode


@dataclass
class CollapseResult:
    mean_probability: float
    uncertainty_width: float
    consensus_score: float
    dominant_driver: str


def leaf_probability(leaf: SimulationNode) -> float:
    if leaf.state is None:
        return 0.5
    return max(0.0, min(1.0, (leaf.state.directional_bias + 1.0) / 2.0))


def reverse_collapse(leaves: Sequence[SimulationNode]) -> CollapseResult:
    if not leaves:
        raise ValueError("reverse_collapse requires at least one leaf")

    probabilities = [leaf_probability(leaf) for leaf in leaves]
    weights = [
        max(
            leaf.probability_weight
            * (0.58 + (0.24 * float(leaf.branch_fitness)) + (0.12 * float(leaf.analog_confidence)) + (0.06 * float(leaf.minority_guardrail))),
            1e-9,
        )
        for leaf in leaves
    ]
    weight_sum = sum(weights)
    normalized = [weight / weight_sum for weight in weights]
    weighted_mean = sum(prob * weight for prob, weight in zip(probabilities, normalized))
    directional_biases = [float((probability - 0.5) * 2.0) for probability in probabilities]
    dispersion = pstdev(probabilities) if len(probabilities) > 1 else 0.0
    weighted_analog = sum(float(leaf.analog_confidence) * weight for leaf, weight in zip(leaves, normalized))
    directional_consensus = abs(sum(bias * weight for bias, weight in zip(directional_biases, normalized)))
    consensus = max(
        0.0,
        min(1.0, (0.54 * (1.0 - min(1.0, dispersion / 0.25))) + (0.28 * directional_consensus) + (0.18 * weighted_analog)),
    )

    dominant_counts = {}
    for leaf in leaves:
        dominant_counts[leaf.dominant_driver] = dominant_counts.get(leaf.dominant_driver, 0.0) + leaf.probability_weight
    dominant_driver = max(dominant_counts, key=dominant_counts.get)

    return CollapseResult(
        mean_probability=round(weighted_mean, 6),
        uncertainty_width=round(min(1.0, dispersion * 4), 6),
        consensus_score=round(consensus, 6),
        dominant_driver=dominant_driver,
    )
