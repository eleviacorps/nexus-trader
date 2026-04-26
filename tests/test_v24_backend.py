import unittest

from src.v24 import HeuristicMetaAggregator, V24EnsembleRiskJudge, build_world_state


class V24BackendTests(unittest.TestCase):
    def test_world_state_builder_exposes_runtime_and_execution_fields(self) -> None:
        state = build_world_state(
            {
                "signal_time_utc": "2024-12-03T10:15:00Z",
                "action": "BUY",
                "reference_close": 2350.0,
                "atr_pct": 0.0011,
                "macro_realized_vol_20": 0.0008,
                "macro_vol_regime_class": 1,
                "return_3": 0.0007,
                "return_12": 0.0014,
                "rr_ratio": 1.9,
                "cabr_score": 0.74,
                "cpm_score": 0.71,
                "confidence_rank": 3,
                "confidence_tier": "high",
                "online_hmm_regime_confidence": 0.68,
                "online_hmm_persistence_count": 4,
            },
            live_performance={"rolling_win_rate_10": 0.6, "consecutive_losses": 1, "daily_pnl": 12.0},
            breaker_state={"trading_allowed": True, "state": "CLEAR"},
            ensemble_state={"risk_score": 0.22, "meta_label_prob": 0.63, "agreement_rate": 0.8, "max_lot": 0.1},
        )
        flat = state.to_flat_features()
        self.assertAlmostEqual(flat["market_structure_rr_ratio"], 1.9, places=6)
        self.assertAlmostEqual(flat["runtime_state_rolling_win_rate_10"], 0.6, places=6)
        self.assertAlmostEqual(flat["execution_context_v22_agreement_rate"], 0.8, places=6)

    def test_meta_aggregator_scores_good_context_as_positive_quality(self) -> None:
        state = build_world_state(
            {
                "signal_time_utc": "2024-12-03T10:15:00Z",
                "action": "BUY",
                "reference_close": 2350.0,
                "atr_pct": 0.0010,
                "macro_realized_vol_20": 0.0008,
                "macro_vol_regime_class": 1,
                "return_3": 0.0008,
                "return_12": 0.0015,
                "rr_ratio": 2.0,
                "cabr_score": 0.78,
                "cpm_score": 0.75,
                "confidence_rank": 4,
                "confidence_tier": "very_high",
                "online_hmm_regime_confidence": 0.72,
                "online_hmm_persistence_count": 4,
            },
            live_performance={"rolling_win_rate_10": 0.7, "consecutive_losses": 0, "daily_pnl": 20.0, "recent_direction_bias": 1.0},
            breaker_state={"trading_allowed": True, "state": "CLEAR"},
            ensemble_state={"risk_score": 0.18, "meta_label_prob": 0.68, "agreement_rate": 0.84, "max_lot": 0.1},
        )
        quality = HeuristicMetaAggregator().predict(state)
        self.assertGreater(quality.expected_rr, 1.5)
        self.assertGreater(quality.profit_probability, 0.55)
        self.assertLess(quality.danger_score, 0.62)

    def test_v24_risk_judge_executes_and_reduces_size_when_needed(self) -> None:
        aggregator = HeuristicMetaAggregator()
        safe_state = build_world_state(
            {
                "signal_time_utc": "2024-12-03T10:15:00Z",
                "action": "BUY",
                "reference_close": 2350.0,
                "atr_pct": 0.0010,
                "macro_vol_regime_class": 1,
                "return_3": 0.0008,
                "return_12": 0.0015,
                "rr_ratio": 2.0,
                "cabr_score": 0.78,
                "cpm_score": 0.75,
                "confidence_rank": 4,
                "confidence_tier": "very_high",
                "online_hmm_regime_confidence": 0.72,
            },
            live_performance={"rolling_win_rate_10": 0.7, "consecutive_losses": 0, "recent_direction_bias": 1.0},
            breaker_state={"trading_allowed": True, "state": "CLEAR"},
            ensemble_state={"risk_score": 0.18, "meta_label_prob": 0.68, "agreement_rate": 0.84, "max_lot": 0.1},
        )
        safe_decision = V24EnsembleRiskJudge().decide(safe_state, aggregator.predict(safe_state))
        self.assertEqual(safe_decision.action, "EXECUTE")

        reduced_state = build_world_state(
            {
                "signal_time_utc": "2024-12-03T10:15:00Z",
                "action": "BUY",
                "reference_close": 2350.0,
                "atr_pct": 0.0012,
                "macro_vol_regime_class": 2,
                "return_3": 0.0005,
                "return_12": 0.0008,
                "rr_ratio": 1.8,
                "cabr_score": 0.70,
                "cpm_score": 0.68,
                "confidence_rank": 3,
                "confidence_tier": "high",
                "online_hmm_regime_confidence": 0.66,
            },
            live_performance={"rolling_win_rate_10": 0.56, "consecutive_losses": 2, "recent_direction_bias": 1.0},
            breaker_state={"trading_allowed": True, "state": "CLEAR"},
            ensemble_state={"risk_score": 0.34, "meta_label_prob": 0.59, "agreement_rate": 0.72, "max_lot": 0.1},
        )
        reduced_decision = V24EnsembleRiskJudge().decide(reduced_state, aggregator.predict(reduced_state))
        self.assertEqual(reduced_decision.action, "REDUCE_SIZE")
        self.assertLess(reduced_decision.size_multiplier, 0.1)

    def test_v24_risk_judge_abstains_when_breaker_or_rr_fail(self) -> None:
        aggregator = HeuristicMetaAggregator()
        blocked_state = build_world_state(
            {
                "signal_time_utc": "2024-12-03T10:15:00Z",
                "action": "SELL",
                "reference_close": 2350.0,
                "atr_pct": 0.0015,
                "macro_vol_regime_class": 3,
                "return_3": -0.0002,
                "return_12": -0.0004,
                "rr_ratio": 1.2,
                "cabr_score": 0.61,
                "cpm_score": 0.58,
                "confidence_rank": 2,
                "confidence_tier": "moderate",
                "online_hmm_regime_confidence": 0.50,
            },
            live_performance={"rolling_win_rate_10": 0.3, "consecutive_losses": 3, "recent_direction_bias": -1.0},
            breaker_state={"trading_allowed": False, "state": "PAUSED"},
            ensemble_state={"risk_score": 0.72, "meta_label_prob": 0.42, "agreement_rate": 0.4, "max_lot": 0.05},
        )
        decision = V24EnsembleRiskJudge().decide(blocked_state, aggregator.predict(blocked_state))
        self.assertEqual(decision.action, "ABSTAIN")
        self.assertIn("v24_", decision.reason)


if __name__ == "__main__":
    unittest.main()
