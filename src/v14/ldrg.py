from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LDRGStatus:
    tier: int
    criteria_met: dict[str, bool]
    blocking_criteria: list[str]
    recommendation: str


def check_ldrg(
    cabr_accuracy: float,
    wf_win_rate: float,
    wf_profitable_months: float,
    wf_months: int,
    stage1_stage2_gap: float,
    s3pta_count: int,
    s3pta_win_rate: Optional[float],
    rsc_max_calibration_error: Optional[float],
    wf_max_monthly_dd: float,
) -> LDRGStatus:
    tier1_criteria = {
        "cabr_above_056": float(cabr_accuracy) >= 0.56,
        "wf_winrate_above_60": float(wf_win_rate) >= 0.60,
        "wf_profitable_months_85pct": float(wf_profitable_months) >= 0.85,
        "wf_months_24plus": int(wf_months) >= 24,
        "stage12_gap_below_5pp": float(stage1_stage2_gap) <= 0.05,
    }
    tier2_criteria = {
        "s3pta_200plus_trades": int(s3pta_count) >= 200,
        "s3pta_winrate_above_55": float(s3pta_win_rate or 0.0) >= 0.55,
        "rsc_error_below_020": float(rsc_max_calibration_error or 1.0) <= 0.20,
        "wf_max_monthly_dd_below_8": float(wf_max_monthly_dd) <= 0.08,
    }
    tier1_met = all(tier1_criteria.values())
    tier2_met = tier1_met and all(tier2_criteria.values())
    current_tier = 0
    if tier1_met:
        current_tier = 1
    if tier2_met:
        current_tier = 2
    all_criteria = {**tier1_criteria, **tier2_criteria}
    blocking = [key for key, value in all_criteria.items() if not value]
    if current_tier == 0:
        recommendation = "Continue research. Tier 1 not yet complete."
    elif current_tier == 1:
        recommendation = "Tier 1 complete. Begin paper trading for Tier 2 validation."
    elif current_tier == 2:
        recommendation = "Tier 2 complete. Ready for live deployment at minimum lot size."
    else:
        recommendation = "All tiers complete."
    return LDRGStatus(
        tier=current_tier,
        criteria_met=all_criteria,
        blocking_criteria=blocking,
        recommendation=recommendation,
    )
