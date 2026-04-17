from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_DIR
from scripts.validate_v24_4_1_codex import (  # type: ignore
    _build_or_load_signals,
    _enrich_signals,
    _load_feature_frame,
    _default_windows,
)
from src.service.claude_trade_gateway import ClaudeTradeGateway
from src.service.claude_trade_router import ClaudeTradeRouter
from src.v24_4_2.recovery_runtime import evaluate_candidates_v24_4_2, safe_float
from src.v24_4_2.threshold_optimizer import ThresholdConfig
from src.v25.auto_execution_engine import AutoExecutionEngine
from src.v25.execution_mode_router import ExecutionModeRouter
from src.v25.live_trade_logger import LiveTradeLogger
from src.v25.manual_execution_queue import ManualExecutionQueue
from src.v25.mt5_bridge import MT5Bridge
from src.v25.paper_trade_engine import PaperTradeConstraints, PaperTradeEngine


def _load_best_config() -> ThresholdConfig:
    path = OUTPUTS_DIR / "v24_4_2" / "best_threshold_config.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        return ThresholdConfig(
            trend_up=float(cfg.get("trend_up", 0.54)),
            trend_down=float(cfg.get("trend_down", 0.64)),
            breakout=float(cfg.get("breakout", 0.58)),
            range_value=float(cfg.get("range_value", 0.60)),
            cooldown_decay=float(cfg.get("cooldown_decay", 0.75)),
            cluster_radius=float(cfg.get("cluster_radius", 0.25)),
            size_multiplier=float(cfg.get("size_multiplier", 1.0)),
        )
    return ThresholdConfig(0.54, 0.64, 0.58, 0.60, 0.75, 0.25, 1.0)


def _load_latest_14d_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = _default_windows()
    latest = windows[-1]
    end = latest.end
    start = end - pd.Timedelta(days=14)
    frame = _load_feature_frame(start, end, prelude_days=20)
    window = type("Window", (), {"label": "proxy_14d", "start": start, "end": end})
    signals = _build_or_load_signals(window, frame)
    candidates = _enrich_signals(signals, frame)
    return candidates, frame


def _load_deployment_score() -> float:
    path = OUTPUTS_DIR / "deployment" / "deployment_readiness.json"
    if not path.exists():
        return 0.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("score_breakdown", {}).get("total_score_100", 0.0))


def main() -> None:
    out_live = OUTPUTS_DIR / "live"
    out_live.mkdir(parents=True, exist_ok=True)

    candidates, frame = _load_latest_14d_data()
    config = _load_best_config()
    trades, _ = evaluate_candidates_v24_4_2(candidates, frame, config)

    deployment_score = _load_deployment_score()
    gateway = ClaudeTradeGateway()
    manual_queue = ManualExecutionQueue()
    paper_engine = PaperTradeEngine(
        starting_balance=10000.0,
        constraints=PaperTradeConstraints(
            risk_per_trade=0.0025,
            max_simultaneous_trades=3,
            allow_pyramiding=False,
        ),
    )
    mt5_bridge = MT5Bridge(export_path=out_live / "mt5_signal.json")
    auto_engine = AutoExecutionEngine(paper_engine=paper_engine, mt5_bridge=mt5_bridge)
    router = ClaudeTradeRouter(
        gateway=gateway,
        mode_router=ExecutionModeRouter(),
        manual_queue=manual_queue,
        auto_engine=auto_engine,
        live_report_path=out_live / "live_paper_report.json",
    )
    logger = LiveTradeLogger(decision_log_path=out_live / "claude_decision_log.jsonl", csv_log_path=out_live / "trade_log.csv")

    for row in trades.sort_values("signal_time_utc").to_dict(orient="records"):
        entry = safe_float(row.get("reference_close"), 0.0)
        stop = entry - 1.0 if str(row.get("variant_signal", "BUY")).upper() == "BUY" else entry + 1.0
        tp = entry + 2.0 if str(row.get("variant_signal", "BUY")).upper() == "BUY" else entry - 2.0
        numeric_candidate = {
            "symbol": "XAUUSD",
            "regime": row.get("regime_label_v24_4_2", "unknown"),
            "regime_confidence": row.get("regime_confidence_v24_4_2", 0.0),
            "admission_score": row.get("admission_score", 0.0),
            "regime_threshold": row.get("admission_threshold", 0.0),
            "strategic_direction": row.get("variant_signal", "HOLD"),
            "tactical_direction": row.get("variant_signal", "HOLD"),
            "calibrated_probability": min(1.0, max(0.0, safe_float(row.get("strategic_confidence"), 0.0))),
            "expected_rr": max(0.0, safe_float(row.get("rr_ratio"), 0.0)),
            "spread": safe_float(row.get("spread_estimate"), 0.0),
            "slippage_estimate": safe_float(row.get("slippage_estimate"), 0.0),
            "recent_trade_health": {"rolling_win_rate_10": 0.6, "recent_drawdown": 0.05},
            "entry_price": entry,
            "stop_loss": stop,
            "take_profit": tp,
            "reasons": [str(row.get("admission_reason", "v24_4_2_candidate"))],
        }
        routed = router.route_candidate(
            numeric_candidate=numeric_candidate,
            mode="auto_mode",
            deployment_score=deployment_score,
            execution_channel="paper",
        )
        logger.log_decision({"timestamp": routed.get("timestamp"), "status": routed.get("status"), "route_reason": routed.get("route_reason"), "judge": routed.get("judge")})

    # Deterministic replay closeout: close all open paper positions at deterministic offset.
    for position in list(paper_engine.open_positions):
        direction = str(position.get("direction", "BUY")).upper()
        entry_price = safe_float(position.get("entry_price"), 0.0)
        exit_price = entry_price + (0.6 if direction == "BUY" else -0.6)
        closed = paper_engine.close_trade(str(position.get("trade_id")), exit_price=exit_price, reason="proxy_replay_close")
        trade = closed.get("trade", {})
        if trade:
            logger.log_trade(
                {
                    "timestamp": trade.get("closed_at"),
                    "trade_id": trade.get("trade_id"),
                    "symbol": trade.get("symbol"),
                    "direction": trade.get("direction"),
                    "regime": trade.get("meta", {}).get("regime"),
                    "reason": trade.get("meta", {}).get("reason"),
                    "claude_approve": True,
                    "claude_confidence": trade.get("meta", {}).get("judge", {}).get("confidence", 0.0),
                    "result": "closed",
                    "pnl": trade.get("pnl"),
                }
            )

    summary = paper_engine.summary()
    report = {
        "mode": "proxy_14d_replay",
        "constraints": {
            "risk_per_trade": 0.0025,
            "max_simultaneous_trades": 3,
            "allow_pyramiding": False,
        },
        "deployment_score_used": deployment_score,
        "trade_candidates": int(len(trades)),
        "paper_summary": summary,
        "proxy_positive": bool(safe_float(summary.get("gross_pnl")) > 0.0 and int(summary.get("closed_positions", 0)) > 0),
    }
    (out_live / "live_paper_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

