from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class MarketWorldState:
    institutional_positioning: float
    retail_sentiment_momentum: float
    structural_memory_strength: float
    crowd_emotional_state: str
    regime_persistence: int
    smart_money_fingerprint: float
    narrative_age: int
    level_memory_hits: int


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V11 persistent world modeling.")


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return float(np.clip(value, lower, upper))


def initial_world_state() -> MarketWorldState:
    return MarketWorldState(
        institutional_positioning=0.0,
        retail_sentiment_momentum=0.0,
        structural_memory_strength=0.0,
        crowd_emotional_state="disbelief",
        regime_persistence=0,
        smart_money_fingerprint=0.0,
        narrative_age=0,
        level_memory_hits=0,
    )


def _sample_summary(frame) -> Any:
    grouped = frame.groupby("sample_id", sort=False)
    summary = grouped.agg(
        timestamp=("timestamp", "first"),
        dominant_regime=("dominant_regime", "first"),
        anchor_price=("anchor_price", "first"),
        branch_direction=("branch_direction", "mean"),
        branch_confidence=("branch_confidence", "mean"),
        macro_consistency_v9=("macro_consistency_v9", "mean"),
        crowd_consistency_v9=("crowd_consistency_v9", "mean"),
        analog_density=("analog_density", "mean"),
        analog_disagreement_v9=("analog_disagreement_v9", "mean"),
        consensus_strength=("consensus_strength", "mean"),
        branch_disagreement=("branch_disagreement", "mean"),
    ).reset_index()
    summary["timestamp"] = pd.to_datetime(summary["timestamp"], utc=True, errors="coerce")
    return summary


def update_world_state(previous: MarketWorldState, row: dict[str, Any], *, crowd_state: str) -> MarketWorldState:
    same_regime = str(row.get("dominant_regime", "")) == crowd_state or previous.crowd_emotional_state == crowd_state
    persistence = previous.regime_persistence + 1 if same_regime else 1
    narrative_age = previous.narrative_age + 1 if previous.crowd_emotional_state == crowd_state else 1
    institutional_flow = (
        float(row.get("macro_consistency_v9", 0.5))
        * float(row.get("branch_direction", 0.0))
        * float(row.get("branch_confidence", 0.5))
    )
    retail_flow = (
        float(row.get("crowd_consistency_v9", 0.5))
        * float(row.get("branch_direction", 0.0))
        * (0.6 + 0.4 * float(row.get("consensus_strength", 0.5)))
    )
    smart_money = (
        0.55 * float(row.get("analog_density", 0.0)) / 24.0
        + 0.25 * float(row.get("macro_consistency_v9", 0.5))
        - 0.20 * float(row.get("analog_disagreement_v9", 0.0))
    )
    return MarketWorldState(
        institutional_positioning=_clip((0.92 * previous.institutional_positioning) + (0.08 * institutional_flow)),
        retail_sentiment_momentum=_clip((0.90 * previous.retail_sentiment_momentum) + (0.10 * retail_flow)),
        structural_memory_strength=float(np.clip((0.88 * previous.structural_memory_strength) + (0.12 * min(float(row.get("branch_disagreement", 0.0)) * 120.0, 1.0)), 0.0, 1.0)),
        crowd_emotional_state=str(crowd_state),
        regime_persistence=int(persistence),
        smart_money_fingerprint=_clip((0.90 * previous.smart_money_fingerprint) + (0.10 * smart_money)),
        narrative_age=int(narrative_age),
        level_memory_hits=int(previous.level_memory_hits),
    )


def roll_world_state_history(frame, crowd_history) -> Any:
    _require_pandas()
    sample_features = _sample_summary(frame)
    merged = sample_features.merge(crowd_history, on="sample_id", how="left")
    merged = merged.sort_values(["timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)
    level_counts: dict[float, int] = {}
    state = initial_world_state()
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        level = round(float(row.get("anchor_price", 0.0)) / 10.0) * 10.0
        level_counts[level] = level_counts.get(level, 0) + 1
        state = update_world_state(state, row, crowd_state=str(row.get("cesm_state", "disbelief")))
        rows.append(
            {
                "sample_id": int(row["sample_id"]),
                "pmwm_institutional_positioning": float(state.institutional_positioning),
                "pmwm_retail_sentiment_momentum": float(state.retail_sentiment_momentum),
                "pmwm_structural_memory_strength": float(np.clip((state.structural_memory_strength + min(level_counts[level] / 5.0, 1.0)) / 2.0, 0.0, 1.0)),
                "pmwm_regime_persistence": int(state.regime_persistence),
                "pmwm_smart_money_fingerprint": float(state.smart_money_fingerprint),
                "pmwm_narrative_age": int(state.narrative_age),
                "pmwm_level_memory_hits": int(level_counts[level]),
                "pmwm_crowd_state": str(state.crowd_emotional_state),
            }
        )
    return pd.DataFrame(rows)


def world_state_to_dict(state: MarketWorldState) -> dict[str, Any]:
    return asdict(state)
