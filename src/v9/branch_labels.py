from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


LABEL_COMPONENT_COLUMNS: tuple[str, ...] = (
    "final_price_accuracy",
    "full_path_similarity",
    "execution_realism",
    "regime_consistency",
    "volatility_realism_v9",
)

LABEL_OUTPUT_COLUMNS: tuple[str, ...] = LABEL_COMPONENT_COLUMNS + (
    "composite_score",
    "top_1_branch",
    "top_3_branches",
    "inside_confidence_cone",
    "minority_rescue_branch",
    "is_top_1_branch",
    "is_top_3_branch",
    "is_minority_rescue_branch",
    "composite_winner_label",
)


@dataclass(frozen=True)
class BranchLabelSummary:
    sample_count: int
    branch_rows: int
    cone_containment_rate: float
    minority_rescue_rate: float
    average_composite_score: float


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V9 branch labels.")


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float32, copy=False), 0.0, 1.0)


def _safe_sign(values: np.ndarray) -> np.ndarray:
    output = np.sign(values.astype(np.float32, copy=False))
    output[output == 0.0] = 1.0
    return output


def _scale_from_frame(frame) -> np.ndarray:
    anchor = frame["anchor_price"].to_numpy(dtype=np.float32)
    actual_final = frame["actual_price_15m"].to_numpy(dtype=np.float32)
    predicted_final = frame["predicted_price_15m"].to_numpy(dtype=np.float32)
    volatility_scale = frame.get("volatility_scale", 1.0)
    if hasattr(volatility_scale, "to_numpy"):
        volatility_scale = volatility_scale.to_numpy(dtype=np.float32)
    else:
        volatility_scale = np.full(len(frame), float(volatility_scale), dtype=np.float32)
    atr_proxy = np.maximum(np.abs(anchor) * np.maximum(volatility_scale, 0.25) * 0.0005, 0.05)
    realized_move = np.abs(actual_final - anchor)
    projected_move = np.abs(predicted_final - anchor)
    return np.maximum.reduce([atr_proxy, realized_move, projected_move, np.full(len(frame), 1e-4, dtype=np.float32)])


def _execution_realism(frame, spread_pips: float, slippage_pips: float, pip_size: float) -> np.ndarray:
    branch_direction = frame["branch_direction"].to_numpy(dtype=np.float32)
    predicted_final = frame["predicted_price_15m"].to_numpy(dtype=np.float32)
    anchor = frame["anchor_price"].to_numpy(dtype=np.float32)
    entry_price = frame["entry_open_price"].to_numpy(dtype=np.float32)
    actual_exit = frame["exit_close_price_15m"].to_numpy(dtype=np.float32)
    predicted_edge = branch_direction * (predicted_final - anchor)
    realized_edge = branch_direction * (actual_exit - entry_price)
    friction = max((spread_pips + slippage_pips) * pip_size, 1e-4)
    net_edge = np.minimum(predicted_edge, realized_edge) - friction
    scaled_edge = np.clip(net_edge / friction, -40.0, 40.0)
    return _clip01(1.0 / (1.0 + np.exp(-scaled_edge)))


def build_branch_labels(
    frame,
    *,
    spread_pips: float = 0.5,
    slippage_pips: float = 0.2,
    pip_size: float = 0.1,
) -> Any:
    _require_pandas()
    labels = frame.copy()
    scale = _scale_from_frame(labels)

    actual_path = labels[["actual_price_5m", "actual_price_10m", "actual_price_15m"]].to_numpy(dtype=np.float32)
    predicted_path = labels[["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"]].to_numpy(dtype=np.float32)
    final_error = np.abs(predicted_path[:, -1] - actual_path[:, -1])
    path_mae = np.mean(np.abs(predicted_path - actual_path), axis=1)

    labels["final_price_accuracy"] = _clip01(np.exp(-(final_error / scale)))
    labels["full_path_similarity"] = _clip01(np.exp(-(path_mae / scale)))
    labels["execution_realism"] = _execution_realism(
        labels,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        pip_size=pip_size,
    )
    labels["regime_consistency"] = _clip01(labels["hmm_regime_match"].to_numpy(dtype=np.float32))
    labels["volatility_realism_v9"] = _clip01(labels["volatility_realism"].to_numpy(dtype=np.float32))
    labels["composite_score"] = (
        0.35 * labels["final_price_accuracy"].to_numpy(dtype=np.float32)
        + 0.30 * labels["full_path_similarity"].to_numpy(dtype=np.float32)
        + 0.15 * labels["execution_realism"].to_numpy(dtype=np.float32)
        + 0.10 * labels["regime_consistency"].to_numpy(dtype=np.float32)
        + 0.10 * labels["volatility_realism_v9"].to_numpy(dtype=np.float32)
    ).astype(np.float32)

    top1_index = labels.groupby("sample_id")["composite_score"].idxmax()
    top1_rows = labels.loc[top1_index, ["sample_id", "branch_id"]].copy()
    top1_rows["branch_id"] = top1_rows["branch_id"].astype(int)
    top1_map = dict(zip(top1_rows["sample_id"].astype(int), top1_rows["branch_id"], strict=False))

    ranking = labels.sort_values(["sample_id", "composite_score"], ascending=[True, False])
    top3_map = (
        ranking.groupby("sample_id", sort=False)
        .head(3)
        .groupby("sample_id")["branch_id"]
        .apply(lambda series: json.dumps([int(value) for value in series.tolist()]))
        .to_dict()
    )

    cone_frame = labels.groupby("sample_id").agg(
        predicted_low_5m=("predicted_price_5m", "min"),
        predicted_high_5m=("predicted_price_5m", "max"),
        predicted_low_10m=("predicted_price_10m", "min"),
        predicted_high_10m=("predicted_price_10m", "max"),
        predicted_low_15m=("predicted_price_15m", "min"),
        predicted_high_15m=("predicted_price_15m", "max"),
        actual_price_5m=("actual_price_5m", "first"),
        actual_price_10m=("actual_price_10m", "first"),
        actual_price_15m=("actual_price_15m", "first"),
    )
    inside_cone = (
        (cone_frame["actual_price_5m"] >= cone_frame["predicted_low_5m"])
        & (cone_frame["actual_price_5m"] <= cone_frame["predicted_high_5m"])
        & (cone_frame["actual_price_10m"] >= cone_frame["predicted_low_10m"])
        & (cone_frame["actual_price_10m"] <= cone_frame["predicted_high_10m"])
        & (cone_frame["actual_price_15m"] >= cone_frame["predicted_low_15m"])
        & (cone_frame["actual_price_15m"] <= cone_frame["predicted_high_15m"])
    )
    inside_cone_map = inside_cone.astype(bool).to_dict()

    labels["_weighted_direction"] = labels["generator_probability"].to_numpy(dtype=np.float32) * labels["branch_direction"].to_numpy(dtype=np.float32)
    consensus_score = labels.groupby("sample_id")["_weighted_direction"].sum()
    actual_direction = labels.groupby("sample_id")["actual_final_return"].first().astype(np.float32).to_numpy()
    actual_direction_series = pd.Series(_safe_sign(actual_direction), index=consensus_score.index)
    consensus_direction_series = pd.Series(
        np.where(consensus_score.to_numpy(dtype=np.float32) >= 0.0, 1.0, -1.0),
        index=consensus_score.index,
    )
    minority_needed = consensus_direction_series != actual_direction_series
    minority_candidates = labels.loc[
        labels["sample_id"].map(actual_direction_series.to_dict()).astype(np.float32) == labels["branch_direction"].to_numpy(dtype=np.float32)
    ].copy()
    minority_candidates = minority_candidates.loc[
        minority_candidates["sample_id"].isin(minority_needed.index[minority_needed])
    ]
    minority_map = (
        minority_candidates.sort_values(["sample_id", "composite_score"], ascending=[True, False])
        .drop_duplicates("sample_id")
        .set_index("sample_id")["branch_id"]
        .astype(int)
        .to_dict()
    )

    top3_pairs = ranking.groupby("sample_id", sort=False).head(3)[["sample_id", "branch_id"]].copy()
    top3_index = pd.MultiIndex.from_frame(top3_pairs.astype({"sample_id": int, "branch_id": int}))

    labels["top_1_branch"] = labels["sample_id"].map(top1_map).fillna(-1).astype(int)
    labels["top_3_branches"] = labels["sample_id"].map(top3_map).fillna("[]")
    labels["inside_confidence_cone"] = labels["sample_id"].map(inside_cone_map).fillna(False).astype(bool)
    labels["minority_rescue_branch"] = labels["sample_id"].map(minority_map).fillna(-1).astype(int)
    labels["is_top_1_branch"] = (labels["branch_id"].astype(int) == labels["top_1_branch"]).astype(np.int8)
    labels["is_top_3_branch"] = pd.MultiIndex.from_frame(labels[["sample_id", "branch_id"]].astype(int)).isin(top3_index).astype(np.int8)
    labels["is_minority_rescue_branch"] = (labels["branch_id"].astype(int) == labels["minority_rescue_branch"]).astype(np.int8)
    labels["composite_winner_label"] = labels["is_top_1_branch"].astype(np.int8)
    labels = labels.drop(columns=["_weighted_direction"])
    return labels


def summarize_branch_labels(frame) -> BranchLabelSummary:
    _require_pandas()
    sample_level = frame.drop_duplicates("sample_id")
    return BranchLabelSummary(
        sample_count=int(frame["sample_id"].nunique()) if len(frame) else 0,
        branch_rows=int(len(frame)),
        cone_containment_rate=float(sample_level["inside_confidence_cone"].mean()) if len(sample_level) else 0.0,
        minority_rescue_rate=float((sample_level["minority_rescue_branch"] >= 0).mean()) if len(sample_level) else 0.0,
        average_composite_score=float(frame["composite_score"].mean()) if len(frame) else 0.0,
    )
