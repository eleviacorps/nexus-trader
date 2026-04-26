from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.v13.cabr import build_cabr_pairs


def difficulty_scores(frame: pd.DataFrame) -> pd.Series:
    actual = pd.to_numeric(frame.get("actual_final_return", 0.0), errors="coerce").fillna(0.0).astype(float)
    predicted = pd.to_numeric(frame.get("predicted_price_15m", 0.0), errors="coerce").fillna(0.0).astype(float)
    anchor = pd.to_numeric(frame.get("anchor_price", 0.0), errors="coerce").fillna(0.0).astype(float)
    path_error = pd.to_numeric(frame.get("path_error", 0.0), errors="coerce").fillna(0.0).abs()
    directional_gap = (predicted - anchor).abs()
    outcome_gap = actual.abs()
    return outcome_gap + directional_gap - path_error


def select_easy_rows(frame: pd.DataFrame, easy_quantile: float = 0.65) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    working["difficulty_score"] = difficulty_scores(working)
    threshold = float(working["difficulty_score"].quantile(float(np.clip(easy_quantile, 0.05, 0.95))))
    easy = working.loc[working["difficulty_score"] >= threshold].copy()
    return easy if not easy.empty else working.copy()


def build_curriculum_pair_payload(
    frame: pd.DataFrame,
    *,
    branch_feature_names: Sequence[str],
    context_feature_names: Sequence[str],
    use_temporal_context: bool = False,
    n_context_bars: int = 12,
    max_pairs: int = 100_000,
    easy_quantile: float = 0.65,
) -> dict[str, dict[str, Any]]:
    easy_rows = select_easy_rows(frame, easy_quantile=easy_quantile)
    easy_payload = build_cabr_pairs(
        easy_rows,
        branch_feature_names=branch_feature_names,
        context_feature_names=context_feature_names,
        use_temporal_context=use_temporal_context,
        n_context_bars=n_context_bars,
        max_pairs=max_pairs,
    )
    full_payload = build_cabr_pairs(
        frame,
        branch_feature_names=branch_feature_names,
        context_feature_names=context_feature_names,
        use_temporal_context=use_temporal_context,
        n_context_bars=n_context_bars,
        max_pairs=max_pairs,
    )
    return {"easy": easy_payload, "full": full_payload}


def build_easy_pair_payload(
    frame: pd.DataFrame,
    *,
    branch_feature_names: Sequence[str],
    context_feature_names: Sequence[str],
    use_temporal_context: bool = False,
    n_context_bars: int = 12,
    max_pairs: int = 100_000,
    easy_quantile: float = 0.65,
) -> dict[str, Any]:
    return build_curriculum_pair_payload(
        frame,
        branch_feature_names=branch_feature_names,
        context_feature_names=context_feature_names,
        use_temporal_context=use_temporal_context,
        n_context_bars=n_context_bars,
        max_pairs=max_pairs,
        easy_quantile=easy_quantile,
    )["easy"]
