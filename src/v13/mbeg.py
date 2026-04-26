from __future__ import annotations


def minority_guard(
    consensus_direction: str,
    minority_direction: str,
    minority_score: float,
    consensus_strength: float,
) -> tuple[bool, float]:
    if str(minority_direction) == str(consensus_direction):
        return True, 1.0
    minority_weight = float(minority_score) * float(1.0 - consensus_strength)
    if minority_weight > 0.35:
        return False, 0.0
    if minority_weight > 0.20:
        return True, 0.50
    return True, 0.85
