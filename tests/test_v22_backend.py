import unittest

import numpy as np
import torch

from src.service.app import build_kimi_context_payload
from src.v22.circuit_breaker import CircuitBreakerConfig, DailyCircuitBreaker
from src.v22.ensemble_judge_stack import EnsembleJudgeStack, LinearMetaLabeler
from src.v22.hybrid_risk_judge import HybridRiskJudge, HybridRiskJudgeConfig
from src.v22.online_hmm import OnlineHMMRegimeDetector, calibrate_confidence_threshold


class V22BackendTests(unittest.TestCase):
    def test_online_hmm_exposes_uncertain_regime_guard(self) -> None:
        detector = OnlineHMMRegimeDetector()
        detector.alpha = np.full(6, 1.0 / 6.0, dtype=np.float64)
        snapshot = detector.summary(direction="SELL", current_price=2500.0, atr_14=5.0)
        self.assertTrue(snapshot["uncertain"])
        self.assertLessEqual(snapshot["lot_size_multiplier"], 0.5)
        self.assertIn("uncertain_regime", snapshot["reasons"])

    def test_circuit_breaker_pauses_after_three_losses(self) -> None:
        breaker = DailyCircuitBreaker()
        breaker.record_trade({"profit": -10.0, "pnl_pct": -0.005})
        breaker.record_trade({"profit": -10.0, "pnl_pct": -0.005})
        status = breaker.record_trade({"profit": -10.0, "pnl_pct": -0.005})
        self.assertFalse(status.trading_allowed)
        self.assertEqual(status.state, "PAUSED")
        self.assertIn("consecutive_losses", status.reasons)

    def test_circuit_breaker_syncs_live_performance_and_relaxed_profile(self) -> None:
        breaker = DailyCircuitBreaker(
            CircuitBreakerConfig(
                arm_after_losses=3,
                consecutive_loss_limit=4,
                min_rolling_win_rate=0.30,
                low_regime_bars_limit=8,
            )
        )
        status = breaker.sync_live_performance(
            {
                "rolling_win_rate_10": 0.20,
                "consecutive_losses": 4,
                "balance": 1000.0,
                "equity": 960.0,
                "daily_pnl": -40.0,
            }
        )
        self.assertEqual(status.consecutive_losses, 4)
        self.assertAlmostEqual(status.rolling_win_rate_10, 0.2, places=6)
        self.assertLessEqual(status.daily_drawdown_pct, -0.02)

    def test_online_hmm_confidence_threshold_calibration_clips_bounds(self) -> None:
        threshold = calibrate_confidence_threshold([0.35, 0.40, 0.55, 0.80], quantile=0.25, minimum=0.52, maximum=0.62)
        self.assertGreaterEqual(threshold, 0.52)
        self.assertLessEqual(threshold, 0.62)

    def test_hybrid_risk_judge_forward_shapes(self) -> None:
        model = HybridRiskJudge(HybridRiskJudgeConfig(series_dim=8, quant_dim=4, hidden_dim=16, num_layers=2))
        outputs = model(torch.randn(2, 12, 8), torch.randn(2, 4))
        self.assertEqual(tuple(outputs["action_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["risk_pred"].shape), (2,))
        self.assertEqual(tuple(outputs["disagree_prob"].shape), (2,))

    def test_ensemble_judge_stack_produces_buy_and_max_lot(self) -> None:
        meta = LinearMetaLabeler(weights=(4.0, -2.0, -3.0, 1.0, -1.0, -1.0), bias=0.0)
        stack = EnsembleJudgeStack(students=[], conformal_quantiles=[0.02, 0.01, -0.03], meta_model=meta)
        outputs = [
            {"action_logits": torch.tensor([3.0, 0.2, -1.0]), "risk_pred": torch.tensor(0.20), "disagree_prob": torch.tensor(0.10)}
            for _ in range(5)
        ]
        prediction = stack.predict_from_outputs(outputs, context_features=[0.6, 0.1])
        self.assertEqual(prediction["action"], "BUY")
        self.assertEqual(prediction["agreement_count"], 5)
        self.assertAlmostEqual(prediction["max_lot"], 0.10, places=6)

    def test_kimi_context_carries_v21_and_v22_runtime(self) -> None:
        context = build_kimi_context_payload(
            {
                "symbol": "XAUUSD",
                "market": {"current_price": 2350.25, "candles": []},
                "simulation": {
                    "direction": "BUY",
                    "overall_confidence": 0.64,
                    "cabr_score": 0.61,
                    "cpm_score": 0.72,
                    "cone_width_pips": 142.0,
                    "confidence_tier": "high",
                    "should_execute": True,
                    "execution_reason": "runtime cleared",
                },
                "technical_analysis": {"structure": "bullish", "location": "discount", "atr_14": 18.2, "order_blocks": []},
                "feeds": {"news": [], "public_discussions": []},
                "paper_trading": {"summary": {"balance": 1000.0, "equity": 1012.5}, "closed_positions": [{"profit": 10.0}, {"profit": -5.0}]},
                "v21_runtime": {
                    "runtime_version": "v21_local",
                    "should_execute": True,
                    "execution_reason": "cleared",
                    "v21_ensemble_prob": 0.66,
                    "v21_meta_label_prob": 0.59,
                    "v21_dangerous_branch_count": 1,
                    "lepl_features": {"kelly_fraction": 0.02, "suggested_lot": 0.05, "conformal_confidence": 0.71},
                },
                "v22_runtime": {
                    "runtime_version": "v22_local",
                    "online_hmm": {"regime_label": "trend_up", "regime_confidence": 0.73, "persistence_count": 3},
                    "circuit_breaker": {"state": "CLEAR", "trading_allowed": True, "rolling_win_rate_10": 0.6},
                    "ensemble": {"action": "BUY", "agreement_rate": 0.8, "meta_label_prob": 0.61, "risk_score": 0.25},
                    "risk_check": {"rr_ratio": 1.8},
                },
            }
        )
        self.assertEqual(context["v21_runtime"]["runtime_version"], "v21_local")
        self.assertAlmostEqual(context["v21_runtime"]["v21_meta_label_prob"], 0.59, places=6)
        self.assertAlmostEqual(context["v22_runtime"]["online_hmm"]["regime_confidence"], 0.73, places=6)
        self.assertAlmostEqual(context["live_performance"]["rolling_win_rate_10"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
