from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class PersonaCalibrationState:
    accuracy_ema: dict[str, float]
    capital_weights: dict[str, float]
    last_timestamp: str | None = None


def _softmax(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    shifted = array - float(array.max(initial=0.0))
    exp = np.exp(shifted)
    weights = exp / max(float(exp.sum()), 1e-6)
    return [float(value) for value in weights.tolist()]


def default_persona_state(personas: Mapping[str, Any]) -> PersonaCalibrationState:
    names = list(personas.keys())
    weights = _softmax([float(getattr(persona, "capital_weight", 1.0)) for persona in personas.values()])
    return PersonaCalibrationState(
        accuracy_ema={name: 0.5 for name in names},
        capital_weights={name: float(weight) for name, weight in zip(names, weights, strict=False)},
        last_timestamp=None,
    )


def aggregate_persona_signals(
    persona_reactions: Mapping[str, list[dict[str, Any]]],
    bots: list[dict[str, Any]],
) -> dict[str, float]:
    bot_probabilities = {str(bot.get("bot_id")): float(bot.get("bullish_probability", 0.5)) for bot in bots}
    persona_signals: dict[str, float] = {}
    for persona, reactions in persona_reactions.items():
        weighted_probs = []
        weights = []
        for reaction in reactions:
            bot_id = str(reaction.get("bot_id", ""))
            weighted_probs.append(bot_probabilities.get(bot_id, 0.5))
            weights.append(max(float(reaction.get("weight", 0.0)), 1e-3))
        if not weighted_probs:
            persona_signals[str(persona)] = 0.0
            continue
        bullish_probability = float(np.average(np.asarray(weighted_probs, dtype=np.float32), weights=np.asarray(weights, dtype=np.float32)))
        persona_signals[str(persona)] = float((bullish_probability - 0.5) * 2.0)
    return persona_signals


def update_persona_calibration(
    state: PersonaCalibrationState,
    persona_signals: Mapping[str, float],
    *,
    actual_direction: float,
    timestamp: str | None = None,
    alpha: float = 0.05,
) -> PersonaCalibrationState:
    names = sorted(set(state.accuracy_ema.keys()) | set(persona_signals.keys()))
    next_accuracy: dict[str, float] = {}
    for name in names:
        previous = float(state.accuracy_ema.get(name, 0.5))
        signal = float(persona_signals.get(name, 0.0))
        predicted_direction = 1.0 if signal >= 0.0 else -1.0
        was_correct = 1.0 if predicted_direction == (1.0 if actual_direction >= 0.0 else -1.0) else 0.0
        next_accuracy[name] = float(alpha * was_correct + (1.0 - alpha) * previous)
    normalized_weights = _softmax([next_accuracy[name] for name in names])
    return PersonaCalibrationState(
        accuracy_ema=next_accuracy,
        capital_weights={name: float(weight) for name, weight in zip(names, normalized_weights, strict=False)},
        last_timestamp=timestamp or state.last_timestamp,
    )


def append_persona_calibration_history(
    path: Path,
    state: PersonaCalibrationState,
    *,
    actual_direction: float | None = None,
) -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for persona calibration history.")
    rows = [
        {
            "timestamp": state.last_timestamp,
            "persona": name,
            "accuracy_ema": float(state.accuracy_ema.get(name, 0.5)),
            "capital_weight": float(state.capital_weights.get(name, 0.0)),
            "actual_direction": None if actual_direction is None else float(actual_direction),
        }
        for name in sorted(state.capital_weights.keys())
    ]
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_parquet(path, index=False)


def load_latest_persona_state(path: Path, personas: Mapping[str, Any]) -> PersonaCalibrationState:
    if pd is None or not path.exists():
        return default_persona_state(personas)
    frame = pd.read_parquet(path)
    if frame.empty:
        return default_persona_state(personas)
    latest_timestamp = frame["timestamp"].dropna().iloc[-1] if frame["timestamp"].notna().any() else None
    latest = frame.loc[frame["timestamp"] == latest_timestamp] if latest_timestamp is not None else frame.tail(len(personas))
    return PersonaCalibrationState(
        accuracy_ema={str(row["persona"]): float(row["accuracy_ema"]) for _, row in latest.iterrows()},
        capital_weights={str(row["persona"]): float(row["capital_weight"]) for _, row in latest.iterrows()},
        last_timestamp=None if latest_timestamp is None else str(latest_timestamp),
    )
