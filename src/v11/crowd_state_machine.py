from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


STATE_TO_ID = {
    "disbelief": 0,
    "greed": 1,
    "euphoria": 2,
    "panic": 3,
    "relief": 4,
}


@dataclass(frozen=True)
class CrowdStateSnapshot:
    state: str
    state_id: int
    confidence: float
    transition_score: float
    greed_pressure: float
    panic_pressure: float


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V11 crowd state modeling.")


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _sample_level_features(frame) -> Any:
    grouped = frame.groupby("sample_id", sort=False)
    summary = grouped.agg(
        timestamp=("timestamp", "first"),
        dominant_regime=("dominant_regime", "first"),
        volatility_scale=("volatility_scale", "mean"),
        branch_disagreement=("branch_disagreement", "mean"),
        consensus_strength=("consensus_strength", "mean"),
        branch_confidence=("branch_confidence", "mean"),
        branch_move_size=("branch_move_size", "mean"),
        analog_disagreement_v9=("analog_disagreement_v9", "mean"),
        crowd_consistency_v9=("crowd_consistency_v9", "mean"),
        news_consistency_v9=("news_consistency_v9", "mean"),
        macro_consistency_v9=("macro_consistency_v9", "mean"),
        minority_share=("branch_direction", lambda values: float(np.mean(np.asarray(values, dtype=np.float32) < 0.0))),
    ).reset_index()
    summary["timestamp"] = pd.to_datetime(summary["timestamp"], utc=True, errors="coerce")
    return summary


def infer_crowd_state(row: dict[str, Any], previous_state: str | None = None) -> CrowdStateSnapshot:
    regime = str(row.get("dominant_regime", "range"))
    vol = float(row.get("volatility_scale", 1.0) or 1.0)
    disagreement = float(row.get("branch_disagreement", 0.0) or 0.0)
    consensus = float(row.get("consensus_strength", 0.5) or 0.5)
    crowd = float(row.get("crowd_consistency_v9", 0.5) or 0.5)
    news = float(row.get("news_consistency_v9", 0.5) or 0.5)
    macro = float(row.get("macro_consistency_v9", 0.5) or 0.5)
    move = float(row.get("branch_move_size", 0.0) or 0.0)

    greed_pressure = _clip01((0.36 * max(move, 0.0) * 120.0) + (0.22 * crowd) + (0.18 * consensus) + (0.12 * macro) + (0.12 * news))
    panic_pressure = _clip01((0.36 * max(-move, 0.0) * 120.0) + (0.24 * vol / 2.0) + (0.18 * disagreement * 120.0) + (0.12 * (1.0 - crowd)) + (0.10 * (1.0 - consensus)))

    if regime == "panic_news_shock" or panic_pressure >= 0.72:
        state = "panic"
        confidence = _clip01(0.55 + 0.35 * panic_pressure)
    elif regime == "bearish_trend" and panic_pressure >= 0.48:
        state = "relief" if previous_state == "panic" and disagreement >= 0.0004 else "panic"
        confidence = _clip01(0.48 + 0.30 * panic_pressure)
    elif regime == "bullish_trend" and greed_pressure >= 0.74 and vol <= 1.3:
        state = "euphoria"
        confidence = _clip01(0.50 + 0.36 * greed_pressure)
    elif regime == "bullish_trend" and greed_pressure >= 0.46:
        state = "greed"
        confidence = _clip01(0.46 + 0.32 * greed_pressure)
    elif previous_state in {"panic", "relief"} and vol >= 1.1 and abs(move) <= 0.001:
        state = "relief"
        confidence = _clip01(0.42 + 0.18 * vol)
    else:
        state = "disbelief"
        confidence = _clip01(0.38 + 0.22 * (1.0 - consensus) + 0.14 * disagreement * 100.0)

    transition_score = _clip01(
        0.45 * abs(greed_pressure - panic_pressure)
        + 0.20 * min(vol / 2.5, 1.0)
        + 0.20 * min(disagreement * 150.0, 1.0)
        + 0.15 * float(previous_state is not None and previous_state != state)
    )
    return CrowdStateSnapshot(
        state=state,
        state_id=int(STATE_TO_ID[state]),
        confidence=confidence,
        transition_score=transition_score,
        greed_pressure=greed_pressure,
        panic_pressure=panic_pressure,
    )


def build_crowd_state_history(frame) -> Any:
    _require_pandas()
    sample_features = _sample_level_features(frame)
    previous_state: str | None = None
    rows: list[dict[str, Any]] = []
    for row in sample_features.sort_values(["timestamp", "sample_id"], kind="mergesort").to_dict(orient="records"):
        snapshot = infer_crowd_state(row, previous_state=previous_state)
        previous_state = snapshot.state
        rows.append(
            {
                "sample_id": int(row["sample_id"]),
                "cesm_state": snapshot.state,
                "cesm_state_id": int(snapshot.state_id),
                "cesm_confidence": float(snapshot.confidence),
                "cesm_transition_score": float(snapshot.transition_score),
                "cesm_greed_pressure": float(snapshot.greed_pressure),
                "cesm_panic_pressure": float(snapshot.panic_pressure),
            }
        )
    return pd.DataFrame(rows)
