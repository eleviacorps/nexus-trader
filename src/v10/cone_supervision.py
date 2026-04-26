from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diversity_loss import diversity_regularized_scores, normalized_path_matrix, pairwise_dispersion

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V10 cone supervision.")


@dataclass(frozen=True)
class ConeSupervisionResult:
    selected_count: int
    objective: float
    cone_containment_rate: float
    cone_width_15m: float
    minority_share: float
    mean_pairwise_dispersion: float


def _containment_score(frame, actual_path: np.ndarray | None) -> float:
    if actual_path is None or len(frame) == 0:
        return 0.0
    paths = frame[["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"]].to_numpy(dtype=np.float32)
    lower = paths.min(axis=0)
    upper = paths.max(axis=0)
    inside = (actual_path >= lower) & (actual_path <= upper)
    return float(np.mean(inside))


def _minority_share(frame) -> float:
    probs = frame["generator_probability"].to_numpy(dtype=np.float32)
    probs = probs / max(float(probs.sum()), 1e-6)
    directions = frame["branch_direction"].to_numpy(dtype=np.float32)
    pos = float(probs[directions >= 0].sum())
    neg = float(probs[directions < 0].sum())
    return min(pos, neg) / max(pos + neg, 1e-6)


def supervise_branch_cone(frame, *, target_branch_count: int, target_width: float, target_minority_share: float, actual_path: np.ndarray | None = None):
    _require_pandas()
    if len(frame) <= target_branch_count:
        selected = frame.copy().reset_index(drop=True)
        containment = _containment_score(selected, actual_path)
        paths = normalized_path_matrix(selected)
        return selected, ConeSupervisionResult(
            selected_count=int(len(selected)),
            objective=float(containment),
            cone_containment_rate=float(containment),
            cone_width_15m=float(paths[:, -1].max(initial=0.0) - paths[:, -1].min(initial=0.0)) if len(paths) else 0.0,
            minority_share=float(_minority_share(selected)),
            mean_pairwise_dispersion=float(pairwise_dispersion(paths)) if len(paths) else 0.0,
        )
    candidate_scores, _ = diversity_regularized_scores(
        frame,
        target_width=target_width,
        target_minority_share=target_minority_share,
    )
    pool = frame.copy().reset_index(drop=True)
    pool["_v10_score"] = candidate_scores
    pool["_terminal_return"] = (
        (pool["predicted_price_15m"].to_numpy(dtype=np.float32) / np.maximum(pool["anchor_price"].to_numpy(dtype=np.float32), 1e-6)) - 1.0
    ).astype(np.float32)
    width_bucket = max(target_width / 3.0, 1e-4)
    pool["_width_bucket"] = np.round(pool["_terminal_return"] / width_bucket).astype(np.int32)
    pool = pool.sort_values(["_v10_score", "branch_confidence", "generator_probability"], ascending=False).reset_index(drop=True)
    chosen: list[int] = []
    seen: set[tuple[int, int]] = set()
    for index, row in pool.iterrows():
        direction = 1 if float(row["branch_direction"]) >= 0.0 else -1
        bucket = int(row["_width_bucket"])
        key = (direction, bucket)
        if key in seen and len(chosen) < max(2, target_branch_count // 2):
            continue
        chosen.append(index)
        seen.add(key)
        if len(chosen) >= target_branch_count:
            break
    if len(chosen) < target_branch_count:
        for index in range(len(pool)):
            if index in chosen:
                continue
            chosen.append(index)
            if len(chosen) >= target_branch_count:
                break
    selected = pool.iloc[chosen].copy().reset_index(drop=True)
    selected["generator_probability"] = (
        selected["_v10_score"].to_numpy(dtype=np.float32)
        / max(float(selected["_v10_score"].sum()), 1e-6)
    ).astype(np.float32)
    selected = selected.drop(columns=["_v10_score", "_terminal_return", "_width_bucket"])
    paths = normalized_path_matrix(selected)
    containment = _containment_score(selected, actual_path)
    return selected, ConeSupervisionResult(
        selected_count=int(len(selected)),
        objective=float(containment + pairwise_dispersion(paths)),
        cone_containment_rate=float(containment),
        cone_width_15m=float(paths[:, -1].max(initial=0.0) - paths[:, -1].min(initial=0.0)),
        minority_share=float(_minority_share(selected)),
        mean_pairwise_dispersion=float(pairwise_dispersion(paths)),
    )
