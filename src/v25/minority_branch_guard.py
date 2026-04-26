from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BranchGuardSelection:
    consensus_branch: dict[str, Any]
    minority_branch: dict[str, Any]
    shock_branch: dict[str, Any]
    cone_width: float
    confidence: float


class MinorityBranchGuard:
    """
    Preserve diversity in final collapse:
    - strongest consensus branch
    - strongest minority branch
    - strongest volatility shock branch
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _direction(branch: Mapping[str, Any]) -> str:
        direction = str(branch.get("direction", branch.get("branch_direction", "HOLD"))).upper()
        if direction in {"BUY", "SELL"}:
            return direction
        move = float(branch.get("branch_move_size", branch.get("trend", 0.0)) or 0.0)
        return "BUY" if move >= 0 else "SELL"

    @staticmethod
    def _score(branch: Mapping[str, Any]) -> float:
        for key in ("blended_rank_score", "branch_quality_score", "cabr_score", "branch_confidence", "score"):
            if key in branch:
                try:
                    return float(branch.get(key))
                except Exception:
                    continue
        return 0.0

    @staticmethod
    def _volatility(branch: Mapping[str, Any]) -> float:
        for key in ("branch_volatility", "volatility_realism", "volatility_scale"):
            try:
                return float(branch.get(key, 0.0))
            except Exception:
                continue
        return 0.0

    @staticmethod
    def _branch_path(branch: Mapping[str, Any]) -> list[float]:
        path = branch.get("path", branch.get("consensus_path", branch.get("future_path", [])))
        if isinstance(path, list):
            values: list[float] = []
            for item in path:
                if isinstance(item, Mapping):
                    value = item.get("price", item.get("value", item.get("target_price", 0.0)))
                else:
                    value = item
                try:
                    values.append(float(value))
                except Exception:
                    values.append(0.0)
            return values
        return []

    def select(self, ranked_branches: Sequence[Mapping[str, Any]]) -> BranchGuardSelection:
        if not ranked_branches:
            empty = {"direction": "HOLD", "path": []}
            return BranchGuardSelection(empty, empty, empty, cone_width=0.0, confidence=0.0)

        branches = [dict(item) for item in ranked_branches]
        consensus = max(branches, key=self._score)
        consensus_direction = self._direction(consensus)
        minority_candidates = [item for item in branches if self._direction(item) != consensus_direction]
        minority = max(minority_candidates, key=self._score) if minority_candidates else dict(consensus)
        shock = max(branches, key=self._volatility)

        # Avoid exact duplicate selection when alternatives exist.
        used_ids = {id(consensus), id(minority)}
        if id(shock) in used_ids:
            alternatives = sorted(branches, key=self._volatility, reverse=True)
            for candidate in alternatives:
                if id(candidate) not in used_ids:
                    shock = candidate
                    break

        consensus_path = self._branch_path(consensus)
        minority_path = self._branch_path(minority)
        shock_path = self._branch_path(shock)
        all_values = np.asarray(consensus_path + minority_path + shock_path, dtype=np.float64) if (consensus_path or minority_path or shock_path) else np.asarray([0.0], dtype=np.float64)
        cone_width = float(np.max(all_values) - np.min(all_values))
        confidence = float(np.clip((self._score(consensus) + self._score(minority) + self._score(shock)) / 3.0, 0.0, 1.0))

        return BranchGuardSelection(
            consensus_branch=dict(consensus),
            minority_branch=dict(minority),
            shock_branch=dict(shock),
            cone_width=cone_width,
            confidence=confidence,
        )

    def build_final_collapse(self, ranked_branches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        selection = self.select(ranked_branches)
        return {
            "consensus_path": self._branch_path(selection.consensus_branch),
            "minority_invalidation_path": self._branch_path(selection.minority_branch),
            "shock_path": self._branch_path(selection.shock_branch),
            "confidence": float(selection.confidence),
            "cone_width": float(selection.cone_width),
            "consensus_direction": self._direction(selection.consensus_branch),
            "minority_direction": self._direction(selection.minority_branch),
            "shock_direction": self._direction(selection.shock_branch),
        }

