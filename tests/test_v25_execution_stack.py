from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.service.claude_trade_gateway import ClaudeGatewayConfig, ClaudeTradeGateway
from src.service.claude_trade_router import ClaudeTradeRouter
from src.v25.auto_execution_engine import AutoExecutionEngine
from src.v25.execution_mode_router import ExecutionCandidate, ExecutionModeRouter
from src.v25.manual_execution_queue import ManualExecutionQueue
from src.v25.mt5_bridge import MT5Bridge
from src.v25.paper_trade_engine import PaperTradeEngine


class V25ExecutionStackTests(unittest.TestCase):
    def test_execution_mode_router_auto_requires_all_gates(self) -> None:
        router = ExecutionModeRouter()
        candidate = ExecutionCandidate(
            direction="BUY",
            confidence=0.7,
            regime="trend_up",
            admission_score=0.69,
            regime_threshold=0.60,
            stop_loss=2299.0,
            take_profit=2304.0,
            reason="test",
            expected_rr=2.0,
        )
        denied = router.route(mode="auto_mode", candidate=candidate, readiness_score=88.0, judge_result={"approve": True})
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.route, "rejected")
        allowed = router.route(mode="auto_mode", candidate=candidate, readiness_score=92.0, judge_result={"approve": True})
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.route, "auto_execute")

    def test_gateway_failover_primary_secondary_then_tertiary(self) -> None:
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "claude_decision_log.jsonl"
            gateway = ClaudeTradeGateway(
                config=ClaudeGatewayConfig(
                    model_order=[
                        "moonshotai/kimi-k2-5",
                        "nvidia/llama-3.3-nemotron-super-49b-v1",
                        "meta/llama-3.1-70b-instruct",
                    ],
                    decision_log_path=log_path,
                )
            )
            calls = []

            def fake_chat_json_request(**kwargs):
                model = kwargs["model"]
                calls.append(model)
                if model != "meta/llama-3.1-70b-instruct":
                    return {"available": False, "error": "simulated_failure"}
                return {"available": True, "content": {"approve": True, "confidence": 0.8, "risk_level": "LOW", "size_multiplier": 1.1, "reason": "ok"}}

            with patch("src.service.claude_trade_gateway._chat_json_request", side_effect=fake_chat_json_request):
                result = gateway.evaluate_candidate(
                    {
                        "regime": "trend_up",
                        "regime_confidence": 0.8,
                        "admission_score": 0.7,
                        "strategic_direction": "BUY",
                        "tactical_direction": "BUY",
                    }
                )
            self.assertTrue(result["approve"])
            self.assertEqual(calls, gateway.config.model_order)

    def test_gateway_cache_fallback_then_fail_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "claude_decision_log.jsonl"
            gateway = ClaudeTradeGateway(
                config=ClaudeGatewayConfig(
                    model_order=["moonshotai/kimi-k2-5"],
                    decision_log_path=log_path,
                    fail_closed=True,
                    cache_fallback_enabled=True,
                )
            )
            # Seed cache with one valid decision.
            log_path.write_text(
                '{"candidate":{"regime":"trend_up","strategic_direction":"BUY","admission_score":0.68,"regime_confidence":0.8},"decision":{"available":true,"source":"live_model","approve":true,"confidence":0.75,"risk_level":"LOW","size_multiplier":1.0,"reason":"seed"}}\n',
                encoding="utf-8",
            )
            with patch("src.service.claude_trade_gateway._chat_json_request", return_value={"available": False, "error": "no_live"}):
                cached = gateway.evaluate_candidate(
                    {
                        "regime": "trend_up",
                        "strategic_direction": "BUY",
                        "admission_score": 0.69,
                        "regime_confidence": 0.79,
                    }
                )
            self.assertTrue(cached["available"])
            self.assertEqual(cached["source"], "cache_fallback")

            gateway_empty = ClaudeTradeGateway(
                config=ClaudeGatewayConfig(
                    model_order=["moonshotai/kimi-k2-5"],
                    decision_log_path=Path(tmp) / "empty_log.jsonl",
                    fail_closed=True,
                    cache_fallback_enabled=True,
                )
            )
            with patch("src.service.claude_trade_gateway._chat_json_request", return_value={"available": False, "error": "no_live"}):
                closed = gateway_empty.evaluate_candidate({"regime": "trend_up", "strategic_direction": "BUY"})
            self.assertFalse(closed["available"])
            self.assertFalse(closed["approve"])
            self.assertEqual(closed["source"], "fail_closed")

    def test_integration_numeric_to_route(self) -> None:
        with TemporaryDirectory() as tmp:
            gateway_log = Path(tmp) / "decision.jsonl"
            gateway = ClaudeTradeGateway(
                config=ClaudeGatewayConfig(
                    model_order=["moonshotai/kimi-k2-5"],
                    decision_log_path=gateway_log,
                    fail_closed=True,
                    cache_fallback_enabled=False,
                )
            )
            with patch(
                "src.service.claude_trade_gateway._chat_json_request",
                return_value={
                    "available": True,
                    "content": {"approve": True, "confidence": 0.8, "risk_level": "LOW", "size_multiplier": 1.0, "reason": "ok"},
                },
            ):
                paper_engine = PaperTradeEngine(starting_balance=10000.0)
                auto = AutoExecutionEngine(paper_engine=paper_engine, mt5_bridge=MT5Bridge(export_path=Path(tmp) / "mt5.json"))
                router = ClaudeTradeRouter(
                    gateway=gateway,
                    mode_router=ExecutionModeRouter(),
                    manual_queue=ManualExecutionQueue(),
                    auto_engine=auto,
                    live_report_path=Path(tmp) / "live_report.json",
                )
                result = router.route_candidate(
                    numeric_candidate={
                        "symbol": "XAUUSD",
                        "regime": "trend_up",
                        "regime_confidence": 0.82,
                        "admission_score": 0.75,
                        "regime_threshold": 0.60,
                        "strategic_direction": "BUY",
                        "tactical_direction": "BUY",
                        "calibrated_probability": 0.72,
                        "expected_rr": 2.0,
                        "entry_price": 2300.0,
                        "stop_loss": 2298.0,
                        "take_profit": 2304.0,
                        "reasons": ["aligned"],
                    },
                    mode="auto_mode",
                    deployment_score=93.0,
                    execution_channel="paper",
                )
            self.assertEqual(result["status"], "auto_executed")


if __name__ == "__main__":
    unittest.main()

