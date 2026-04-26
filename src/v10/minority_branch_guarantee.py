from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V10 minority branch handling.")


@dataclass(frozen=True)
class MinorityGuaranteeResult:
    branch_rows: int
    minority_share: float
    rescued: bool


def _normalize_probabilities(frame):
    values = frame["generator_probability"].to_numpy(dtype=np.float32)
    total = max(float(values.sum()), 1e-6)
    frame["generator_probability"] = (values / total).astype(np.float32)
    return frame


def _minority_share(frame) -> float:
    probs = frame["generator_probability"].to_numpy(dtype=np.float32)
    probs = probs / max(float(probs.sum()), 1e-6)
    directions = frame["branch_direction"].to_numpy(dtype=np.float32)
    pos = float(probs[directions >= 0].sum())
    neg = float(probs[directions < 0].sum())
    total = max(pos + neg, 1e-6)
    return min(pos, neg) / total


def _mirror_branch_row(row):
    anchor = float(row["anchor_price"])
    mirrored = row.copy()
    for column in ["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"]:
        mirrored[column] = anchor - (float(row[column]) - anchor) * 0.92
    mirrored["branch_direction"] = -1.0 * float(row.get("branch_direction", 1.0))
    mirrored["branch_move_size"] = (float(mirrored["predicted_price_15m"]) / max(anchor, 1e-6)) - 1.0
    mirrored["branch_move_zscore"] = -1.0 * float(row.get("branch_move_zscore", 0.0)) * 0.92
    mirrored["vwap_distance"] = -1.0 * float(row.get("vwap_distance", 0.0))
    mirrored["news_consistency"] = float(np.clip(1.0 - float(row.get("news_consistency", 0.5)), 0.0, 1.0))
    mirrored["crowd_consistency"] = float(np.clip(1.0 - float(row.get("crowd_consistency", 0.5)), 0.0, 1.0))
    mirrored["macro_alignment"] = float(np.clip(1.0 - float(row.get("macro_alignment", 0.5)), 0.0, 1.0))
    mirrored["branch_label"] = "minority_rescue"
    mirrored["leaf_branch_label"] = "minority_rescue"
    mirrored["leaf_minority_guardrail"] = float(np.clip(max(float(row.get("leaf_minority_guardrail", 0.0)), 0.85), 0.0, 1.0))
    mirrored["branch_confidence"] = float(np.clip(float(row.get("branch_confidence", 0.0)) * 0.82, 0.0, 1.0))
    mirrored["generator_probability"] = float(max(float(row.get("generator_probability", 0.0)) * 0.35, 1e-4))
    mirrored["winning_branch"] = False
    mirrored["winner_label"] = 0
    mirrored["v10_minority_rescue"] = True
    return mirrored


def enforce_minority_branch_guarantee(frame, *, target_share: float):
    _require_pandas()
    if len(frame) == 0:
        return frame, MinorityGuaranteeResult(0, 0.0, False)
    output = frame.copy()
    output["v10_minority_rescue"] = output.get("v10_minority_rescue", False)
    output = _normalize_probabilities(output)
    current_share = _minority_share(output)
    rescued = False
    if current_share + 1e-6 < target_share:
        dominant_direction = 1.0 if output["generator_probability"].where(output["branch_direction"] >= 0.0, 0.0).sum() >= output["generator_probability"].where(output["branch_direction"] < 0.0, 0.0).sum() else -1.0
        minority_rows = output.loc[output["branch_direction"] != dominant_direction].copy()
        if minority_rows.empty:
            source = output.sort_values(["generator_probability", "branch_confidence"], ascending=False).iloc[0]
            minority_rows = pd.DataFrame([_mirror_branch_row(source)])
        else:
            minority_rows["generator_probability"] = minority_rows["generator_probability"].astype(np.float32) * np.float32(1.5)
        output = pd.concat([output, minority_rows], ignore_index=True)
        rescued = True
    output = _normalize_probabilities(output)
    return output, MinorityGuaranteeResult(
        branch_rows=int(len(output)),
        minority_share=float(_minority_share(output)),
        rescued=rescued,
    )
