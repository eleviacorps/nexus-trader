from __future__ import annotations

import unittest

from src.v20.runtime import V20RuntimeBundle, _build_branch_candidates, _direction_signal, build_v20_local_judge


class V20RuntimeTests(unittest.TestCase):
    def test_direction_signal_respects_row_inputs(self) -> None:
        row = {
            "macro_trend_strength": 0.3,
            "quant_trend_score": 0.2,
            "rsi_14": 58.0,
            "hurst_overall": 0.57,
            "mfg_mean_belief": 0.001,
            "roc_15m": 0.002,
        }
        signal = _direction_signal(row)
        self.assertGreater(signal, 0.0)

    def test_branch_candidates_score_and_sort(self) -> None:
        candidates = _build_branch_candidates(
            {
                "close": 2300.0,
                "atr_pct": 0.0012,
                "macro_trend_strength": 0.4,
                "quant_trend_score": 0.2,
                "rsi_14": 55.0,
                "hurst_overall": 0.58,
                "mfg_mean_belief": 0.001,
                "roc_15m": 0.001,
                "quant_regime_strength": 0.6,
                "quant_route_confidence": 0.6,
                "quant_vol_realism": 0.7,
                "hmm_state": 1,
                "mfg_disagreement": 0.2,
            }
        )
        self.assertEqual(len(candidates), 64)
        self.assertIn("v20_cabr_score", candidates.columns)

    def test_local_judge_formats_trade_fields(self) -> None:
        bundle = V20RuntimeBundle(
            feature_row={
                "close": 2300.0,
                "direction_signal": 0.25,
                "atr_pct": 0.002,
                "hmm_state": 1,
                "hmm_state_name": "trending_up",
                "hurst_overall": 0.6,
                "mfg_disagreement": 0.2,
            },
            branches=_build_branch_candidates(
                {
                    "close": 2300.0,
                    "atr_pct": 0.0012,
                    "macro_trend_strength": 0.4,
                    "quant_trend_score": 0.2,
                    "rsi_14": 55.0,
                    "hurst_overall": 0.58,
                    "mfg_mean_belief": 0.001,
                    "roc_15m": 0.001,
                    "quant_regime_strength": 0.6,
                    "quant_route_confidence": 0.6,
                    "quant_vol_realism": 0.7,
                    "hmm_state": 1,
                    "mfg_disagreement": 0.2,
                }
            ),
            consensus_path=[2300.0, 2301.0, 2302.0],
            minority_path=[2300.0, 2299.5, 2299.0],
            cone_upper=[2300.5, 2301.5, 2302.5],
            cone_lower=[2299.5, 2300.5, 2301.5],
            confidence=0.85,
            regime_probs=[0.1, 0.6, 0.1, 0.1, 0.05, 0.05],
        )
        judge = build_v20_local_judge({}, bundle)
        self.assertTrue(judge["available"])
        self.assertIn(judge["content"]["final_call"], {"BUY", "SELL", "SKIP"})


if __name__ == "__main__":
    unittest.main()
