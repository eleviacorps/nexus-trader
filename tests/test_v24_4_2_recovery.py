from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from src.v24_4_2.adaptive_admission import AdaptiveAdmission, AdmissionCandidate
from src.v24_4_2.regime_threshold_router import RegimeContext, RegimeThresholdRouter
from src.v24_4_2.sell_bias_guard import MarketExecutionContext, SellBiasGuard, StreakContext


class V24_4_2RecoveryTests(unittest.TestCase):
    def test_regime_router_handles_chop_and_unknown(self) -> None:
        router = RegimeThresholdRouter()
        chop = router.route(RegimeContext("chop", regime_confidence=0.8, rolling_expectancy=0.0, rolling_win_rate=0.5))
        unknown = router.route(RegimeContext("unknown", regime_confidence=0.5, rolling_expectancy=0.0, rolling_win_rate=0.5))
        self.assertFalse(chop.enabled)
        self.assertTrue(unknown.enabled)
        self.assertTrue(unknown.strategic_only)

    def test_admission_formula_matches_spec(self) -> None:
        admission = AdaptiveAdmission(router=RegimeThresholdRouter())
        candidate = AdmissionCandidate(
            calibrated_probability=0.70,
            tactical_cabr=0.60,
            regime_profitability=0.55,
            execution_quality=0.80,
            strategic_alignment=1.00,
            recent_trade_health=0.65,
            strategic_direction="BUY",
            tactical_direction="BUY",
        )
        score = admission.score(candidate)
        expected = (0.35 * 0.70) + (0.20 * 0.60) + (0.15 * 0.55) + (0.10 * 0.80) + (0.10 * 1.00) + (0.10 * 0.65)
        self.assertAlmostEqual(score, expected, places=8)

    def test_sell_guard_enforces_stricter_sell_threshold(self) -> None:
        guard = SellBiasGuard()
        now = datetime.now(tz=UTC)
        decision = guard.evaluate_sell(
            candidate={
                "direction": "SELL",
                "regime_confidence": 0.8,
                "tactical_cabr": 0.75,
                "admission_score": 0.40,
                "timestamp": now,
                "entry_price": 2300.0,
            },
            market_exec_ctx=MarketExecutionContext(
                spread=0.10,
                slippage_estimate=0.03,
                buy_threshold=0.56,
            ),
            streak_ctx=StreakContext(
                consecutive_sell_trades=0,
                consecutive_sell_losses=0,
                cooldown_bars=0,
                recent_sell_entries=(),
            ),
        )
        self.assertFalse(decision.allow)
        self.assertEqual(decision.reason, "sell_score_below_buy_threshold_plus_buffer")

    def test_sell_guard_cooldown_escalates_after_losses(self) -> None:
        guard = SellBiasGuard()
        now = datetime.now(tz=UTC)
        decision = guard.evaluate_sell(
            candidate={
                "direction": "SELL",
                "regime_confidence": 0.8,
                "tactical_cabr": 0.8,
                "admission_score": 0.82,
                "timestamp": now,
                "entry_price": 2300.0,
            },
            market_exec_ctx=MarketExecutionContext(spread=0.10, slippage_estimate=0.03, buy_threshold=0.60),
            streak_ctx=StreakContext(
                consecutive_sell_trades=1,
                consecutive_sell_losses=2,
                cooldown_bars=1,
                recent_sell_entries=(),
            ),
        )
        self.assertTrue(decision.allow)
        self.assertGreaterEqual(decision.adjusted_cooldown_bars, 4)

    def test_sell_cluster_duplicate_block_rejects(self) -> None:
        guard = SellBiasGuard(duplicate_cluster_radius=0.25, duplicate_cluster_minutes=10)
        now = datetime.now(tz=UTC)
        decision = guard.evaluate_sell(
            candidate={
                "direction": "SELL",
                "regime_confidence": 0.9,
                "tactical_cabr": 0.9,
                "admission_score": 0.9,
                "timestamp": now,
                "entry_price": 2300.10,
            },
            market_exec_ctx=MarketExecutionContext(spread=0.05, slippage_estimate=0.02, buy_threshold=0.60),
            streak_ctx=StreakContext(
                consecutive_sell_trades=0,
                consecutive_sell_losses=0,
                cooldown_bars=0,
                recent_sell_entries=((now - timedelta(minutes=3), 2300.00),),
            ),
        )
        self.assertFalse(decision.allow)
        self.assertEqual(decision.reason, "sell_cluster_duplicate_blocked")


if __name__ == "__main__":
    unittest.main()
