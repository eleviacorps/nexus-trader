from __future__ import annotations

import unittest

from src.v14.ldrg import check_ldrg


class TestV14LDRG(unittest.TestCase):
    def test_tier_zero_when_cabr_short(self) -> None:
        status = check_ldrg(
            cabr_accuracy=0.55,
            wf_win_rate=0.61,
            wf_profitable_months=0.90,
            wf_months=30,
            stage1_stage2_gap=0.01,
            s3pta_count=10,
            s3pta_win_rate=0.50,
            rsc_max_calibration_error=0.30,
            wf_max_monthly_dd=0.05,
        )
        self.assertEqual(status.tier, 0)

    def test_tier_one_when_research_criteria_pass(self) -> None:
        status = check_ldrg(
            cabr_accuracy=0.57,
            wf_win_rate=0.61,
            wf_profitable_months=0.90,
            wf_months=30,
            stage1_stage2_gap=0.01,
            s3pta_count=50,
            s3pta_win_rate=0.50,
            rsc_max_calibration_error=0.30,
            wf_max_monthly_dd=0.05,
        )
        self.assertEqual(status.tier, 1)


if __name__ == "__main__":
    unittest.main()
