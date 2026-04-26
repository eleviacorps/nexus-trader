from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiversityLossBreakdown:
    mean_pairwise_dispersion: float
    minority_share: float
    cone_width_15m: float
    target_width_score: float
    diversity_loss: float


def normalized_path_matrix(frame) -> np.ndarray:
    anchor = np.maximum(frame["anchor_price"].to_numpy(dtype=np.float32).reshape(-1, 1), 1e-6)
    path = frame[["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"]].to_numpy(dtype=np.float32)
    return (path - anchor) / anchor


def pairwise_dispersion(paths: np.ndarray) -> float:
    if len(paths) <= 1:
        return 0.0
    total = 0.0
    count = 0
    for left in range(len(paths)):
        deltas = paths[left + 1 :] - paths[left]
        if deltas.size == 0:
            continue
        distances = np.sqrt(np.sum(np.square(deltas), axis=1))
        total += float(distances.sum())
        count += int(len(distances))
    return total / max(count, 1)


def minority_share(directions: np.ndarray, weights: np.ndarray) -> float:
    if len(directions) == 0:
        return 0.0
    positive = float(weights[directions >= 0].sum())
    negative = float(weights[directions < 0].sum())
    total = max(positive + negative, 1e-6)
    return min(positive, negative) / total


def diversity_regularized_scores(frame, *, target_width: float, target_minority_share: float) -> tuple[np.ndarray, DiversityLossBreakdown]:
    if len(frame) == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, DiversityLossBreakdown(0.0, 0.0, 0.0, 0.0, 1.0)
    paths = normalized_path_matrix(frame)
    terminal = paths[:, -1]
    weights = frame["generator_probability"].to_numpy(dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-6)
    directions = np.sign(np.where(np.abs(terminal) <= 1e-6, frame["branch_direction"].to_numpy(dtype=np.float32), terminal))
    base_score = (
        0.42 * frame["generator_probability"].to_numpy(dtype=np.float32)
        + 0.18 * frame["branch_confidence"].to_numpy(dtype=np.float32)
        + 0.12 * frame["volatility_realism"].to_numpy(dtype=np.float32)
        + 0.12 * frame["hmm_regime_match"].to_numpy(dtype=np.float32)
        + 0.08 * frame["analog_similarity"].to_numpy(dtype=np.float32)
        + 0.08 * frame["leaf_minority_guardrail"].to_numpy(dtype=np.float32)
    )
    pairwise = pairwise_dispersion(paths)
    novelty = np.mean(np.abs(paths - np.median(paths, axis=0, keepdims=True)), axis=1)
    width_15m = float(max(terminal.max(initial=0.0) - terminal.min(initial=0.0), 0.0))
    width_score = float(np.exp(-abs(width_15m - target_width) / max(target_width, 1e-6)))
    minority = minority_share(directions, weights)
    minority_need = max(target_minority_share - minority, 0.0)
    dominant_direction = 1.0 if np.average(directions, weights=weights) >= 0.0 else -1.0
    minority_bonus = np.where(directions == dominant_direction, 0.0, 0.35 + 0.65 * minority_need).astype(np.float32)
    scores = base_score + novelty.astype(np.float32) + minority_bonus
    scores = scores * np.float32(0.85 + 0.15 * width_score)
    spread_score = float(np.tanh(pairwise * 22.0))
    diversity_loss = float(
        1.0
        - np.clip(
            (0.45 * spread_score)
            + (0.25 * min(1.0, minority / max(target_minority_share, 1e-6)))
            + (0.30 * width_score),
            0.0,
            1.0,
        )
    )
    return scores.astype(np.float32), DiversityLossBreakdown(
        mean_pairwise_dispersion=float(pairwise),
        minority_share=float(minority),
        cone_width_15m=width_15m,
        target_width_score=width_score,
        diversity_loss=diversity_loss,
    )
