from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .cone_supervision import supervise_branch_cone
from .diversity_loss import diversity_regularized_scores
from .minority_branch_guarantee import enforce_minority_branch_guarantee
from .regime_conditioned_generator import GenerationRegimeProfile, infer_generation_regime, temperature_schedule

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V10 diversified generation.")


@dataclass(frozen=True)
class DiversifiedSampleReport:
    sample_id: int
    source_rows: int
    candidate_rows: int
    selected_rows: int
    regime: str
    temperatures: tuple[float, ...]
    minority_rescued: bool
    minority_share: float
    cone_containment_rate: float
    cone_width_15m: float
    mean_pairwise_dispersion: float


def _clone_with_temperature(sample, *, temperature: float, mode: str, profile: GenerationRegimeProfile):
    rows = sample.copy()
    anchor = rows["anchor_price"].to_numpy(dtype=np.float32)
    base_probs = rows["generator_probability"].to_numpy(dtype=np.float32)
    for horizon_index, column in enumerate(["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"], start=1):
        price = rows[column].to_numpy(dtype=np.float32)
        delta = price - anchor
        horizon_scale = 0.90 + 0.08 * horizon_index
        adjusted = anchor + (delta * temperature * horizon_scale)
        if mode == "minority":
            adjusted = anchor - (delta * (0.86 + 0.04 * horizon_index))
        elif mode == "balanced":
            adjusted = anchor + (delta * (0.96 + 0.02 * horizon_index) * min(temperature, 1.1))
        rows[column] = adjusted.astype(np.float32)
    rows["branch_move_size"] = ((rows["predicted_price_15m"].to_numpy(dtype=np.float32) / np.maximum(anchor, 1e-6)) - 1.0).astype(np.float32)
    rows["branch_direction"] = np.where(rows["branch_move_size"].to_numpy(dtype=np.float32) >= 0.0, 1.0, -1.0).astype(np.float32)
    rows["branch_move_zscore"] = (rows["branch_move_zscore"].to_numpy(dtype=np.float32) * np.float32(temperature if mode != "minority" else -0.92)).astype(np.float32)
    rows["branch_volatility"] = (rows["branch_volatility"].to_numpy(dtype=np.float32) * np.float32(max(0.65, temperature))).astype(np.float32)
    rows["atr_normalized_move"] = (rows["atr_normalized_move"].to_numpy(dtype=np.float32) * np.float32(max(0.65, temperature))).astype(np.float32)
    realism_scale = np.exp(-abs(float(temperature) - 1.0) * (0.42 + 0.28 * profile.transition_risk))
    rows["volatility_realism"] = np.clip(rows["volatility_realism"].to_numpy(dtype=np.float32) * np.float32(realism_scale), 0.0, 1.0)
    rows["generator_probability"] = (base_probs * np.float32(np.exp(-abs(float(temperature) - 1.0) * 0.35))).astype(np.float32)
    rows["branch_entropy"] = (-rows["generator_probability"].to_numpy(dtype=np.float32) * np.log(np.maximum(rows["generator_probability"].to_numpy(dtype=np.float32), 1e-6))).astype(np.float32)
    rows["leaf_branch_label"] = rows["leaf_branch_label"].astype(str) + f"|{mode}|t{temperature:.2f}"
    rows["v10_generation_mode"] = mode
    rows["v10_temperature"] = np.float32(temperature)
    rows["v10_regime_label"] = profile.label
    return rows


def diversify_archive_sample(sample):
    _require_pandas()
    source = sample.copy().reset_index(drop=True)
    if len(source) == 0:
        return source, DiversifiedSampleReport(0, 0, 0, 0, "unknown", tuple(), False, 0.0, 0.0, 0.0, 0.0)
    context = source.iloc[0].to_dict()
    profile = infer_generation_regime(context)
    schedule = temperature_schedule(profile)
    candidates = [source.assign(v10_generation_mode="source", v10_temperature=np.float32(1.0), v10_regime_label=profile.label)]
    ranked = source.sort_values(["generator_probability", "branch_confidence"], ascending=False)
    seed_rows = ranked.head(max(4, min(len(ranked), (profile.target_branch_count // 2) + 1)))
    for temperature in schedule:
        candidates.append(_clone_with_temperature(seed_rows, temperature=float(temperature), mode="balanced", profile=profile))
        candidates.append(_clone_with_temperature(seed_rows, temperature=float(temperature), mode="consensus", profile=profile))
    candidates.append(_clone_with_temperature(seed_rows.head(max(2, len(seed_rows) // 2)), temperature=float(profile.temperature_ceiling), mode="minority", profile=profile))
    candidate_frame = pd.concat(candidates, ignore_index=True)
    candidate_frame["branch_id"] = np.arange(1, len(candidate_frame) + 1, dtype=np.int32)
    candidate_frame, minority_result = enforce_minority_branch_guarantee(
        candidate_frame,
        target_share=profile.minority_target_share,
    )
    actual_path = source[["actual_price_5m", "actual_price_10m", "actual_price_15m"]].iloc[0].to_numpy(dtype=np.float32)
    selected, cone_result = supervise_branch_cone(
        candidate_frame,
        target_branch_count=profile.target_branch_count,
        target_width=profile.cone_width_target,
        target_minority_share=profile.minority_target_share,
        actual_path=actual_path,
    )
    selected, minority_result = enforce_minority_branch_guarantee(selected, target_share=profile.minority_target_share)
    selected["sample_id"] = source["sample_id"].iloc[0]
    selected["timestamp"] = source["timestamp"].iloc[0]
    selected["dominant_regime"] = source["dominant_regime"].iloc[0]
    selected["branch_id"] = np.arange(1, len(selected) + 1, dtype=np.int32)
    path_error = (
        np.mean(
            np.abs(
                selected[["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"]].to_numpy(dtype=np.float32)
                - actual_path.reshape(1, -1)
            ),
            axis=1,
        )
        / max(float(source["anchor_price"].iloc[0]), 1e-6)
    )
    selected["winner_label"] = 0
    selected["winning_branch"] = False
    if len(selected):
        winner = int(np.argmin(path_error))
        selected.loc[winner, "winner_label"] = 1
        selected.loc[winner, "winning_branch"] = True
    selected_scores, _ = diversity_regularized_scores(
        selected,
        target_width=profile.cone_width_target,
        target_minority_share=profile.minority_target_share,
    )
    selected["v10_diversity_score"] = selected_scores.astype(np.float32)
    selected["generator_probability"] = (selected_scores / max(float(selected_scores.sum()), 1e-6)).astype(np.float32)
    report = DiversifiedSampleReport(
        sample_id=int(source["sample_id"].iloc[0]),
        source_rows=int(len(source)),
        candidate_rows=int(len(candidate_frame)),
        selected_rows=int(len(selected)),
        regime=profile.label,
        temperatures=tuple(float(value) for value in schedule),
        minority_rescued=bool(minority_result.rescued),
        minority_share=float(minority_result.minority_share),
        cone_containment_rate=float(cone_result.cone_containment_rate),
        cone_width_15m=float(cone_result.cone_width_15m),
        mean_pairwise_dispersion=float(cone_result.mean_pairwise_dispersion),
    )
    return selected.reset_index(drop=True), report


def diversify_branch_archive(frame):
    _require_pandas()
    outputs = []
    reports = []
    for _, sample in frame.groupby("sample_id", sort=False):
        diversified, report = diversify_archive_sample(sample)
        outputs.append(diversified)
        reports.append(asdict(report))
    if not outputs:
        return frame.iloc[0:0].copy(), []
    return pd.concat(outputs, ignore_index=True), reports
