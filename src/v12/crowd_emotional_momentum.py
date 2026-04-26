from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.v11.crowd_state_machine import STATE_TO_ID, build_crowd_state_history


@dataclass(frozen=True)
class CrowdEmotionalState:
    state: str
    momentum: float
    fragility: float
    conviction: float
    narrative_age: int


def compute_emotional_momentum(state_history: list[float], window: int = 5) -> float:
    if len(state_history) < window + 1:
        return 0.0
    recent = np.asarray(state_history[-window:], dtype=np.float32)
    return float(np.polyfit(np.arange(window, dtype=np.float32), recent, 1)[0])


def build_crowd_emotional_momentum(frame: pd.DataFrame) -> pd.DataFrame:
    crowd = build_crowd_state_history(frame)
    crowd = crowd.sort_values("sample_id").reset_index(drop=True)
    state_intensity_history: list[float] = []
    narrative_age = 0
    previous_regime = None
    rows: list[dict[str, float | str | int]] = []

    for row in crowd.itertuples(index=False):
        state = str(row.cesm_state)
        confidence = float(row.cesm_confidence)
        state_intensity = confidence * float(STATE_TO_ID.get(state, 0))
        state_intensity_history.append(state_intensity)
        momentum = compute_emotional_momentum(state_intensity_history)
        fragility = float(np.clip(1.0 - abs(confidence - 0.5) * 2.0, 0.0, 1.0))
        conviction = float(np.clip(0.5 + (0.35 * confidence) - (0.25 * float(row.cesm_transition_score)), 0.0, 1.0))
        current_regime = state
        if previous_regime is None or current_regime == previous_regime:
            narrative_age += 1
        else:
            narrative_age = 0
        previous_regime = current_regime
        rows.append(
            {
                "sample_id": int(row.sample_id),
                "cem_state": state,
                "cem_momentum": momentum,
                "cem_fragility": fragility,
                "cem_conviction": conviction,
                "cem_narrative_age": int(narrative_age),
            }
        )

    return pd.DataFrame(rows)
