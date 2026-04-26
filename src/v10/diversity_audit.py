from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .diversity_loss import normalized_path_matrix, pairwise_dispersion

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V10 diversity audit.")


@dataclass(frozen=True)
class DiversityAuditSummary:
    sample_count: int
    branch_rows: int
    mean_consensus_strength: float
    mean_minority_share: float
    mean_direction_std: float
    mean_pairwise_dispersion: float
    mean_cone_width_15m: float
    cone_containment_rate: float
    full_path_containment_rate: float


def audit_branch_archive(frame) -> DiversityAuditSummary:
    _require_pandas()
    if len(frame) == 0:
        return DiversityAuditSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    consensus_strength = []
    minority_share = []
    direction_std = []
    pairwise = []
    widths = []
    containment_steps = []
    containment_full = []
    for _, sample in frame.groupby("sample_id", sort=False):
        probs = sample["generator_probability"].to_numpy(dtype=np.float32)
        probs = probs / max(float(probs.sum()), 1e-6)
        directions = sample["branch_direction"].to_numpy(dtype=np.float32)
        pos = float(probs[directions >= 0].sum())
        neg = float(probs[directions < 0].sum())
        consensus_strength.append(max(pos, neg))
        minority_share.append(min(pos, neg))
        direction_std.append(float(np.std(directions)))
        paths = normalized_path_matrix(sample)
        pairwise.append(float(pairwise_dispersion(paths)))
        widths.append(float(paths[:, -1].max(initial=0.0) - paths[:, -1].min(initial=0.0)))
        actual = sample[["actual_price_5m", "actual_price_10m", "actual_price_15m"]].iloc[0].to_numpy(dtype=np.float32)
        predicted = sample[["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"]].to_numpy(dtype=np.float32)
        lower = predicted.min(axis=0)
        upper = predicted.max(axis=0)
        inside = (actual >= lower) & (actual <= upper)
        containment_steps.append(float(inside.mean()))
        containment_full.append(float(inside.all()))
    return DiversityAuditSummary(
        sample_count=int(frame["sample_id"].nunique()),
        branch_rows=int(len(frame)),
        mean_consensus_strength=float(np.mean(consensus_strength)),
        mean_minority_share=float(np.mean(minority_share)),
        mean_direction_std=float(np.mean(direction_std)),
        mean_pairwise_dispersion=float(np.mean(pairwise)),
        mean_cone_width_15m=float(np.mean(widths)),
        cone_containment_rate=float(np.mean(containment_steps)),
        full_path_containment_rate=float(np.mean(containment_full)),
    )


def audit_to_dict(frame) -> dict[str, float | int]:
    return asdict(audit_branch_archive(frame))


def render_audit_markdown(title: str, summary: DiversityAuditSummary) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- sample_count: `{summary.sample_count}`",
            f"- branch_rows: `{summary.branch_rows}`",
            f"- mean_consensus_strength: `{summary.mean_consensus_strength:.6f}`",
            f"- mean_minority_share: `{summary.mean_minority_share:.6f}`",
            f"- mean_direction_std: `{summary.mean_direction_std:.6f}`",
            f"- mean_pairwise_dispersion: `{summary.mean_pairwise_dispersion:.6f}`",
            f"- mean_cone_width_15m: `{summary.mean_cone_width_15m:.6f}`",
            f"- cone_containment_rate: `{summary.cone_containment_rate:.6f}`",
            f"- full_path_containment_rate: `{summary.full_path_containment_rate:.6f}`",
            "",
        ]
    )
