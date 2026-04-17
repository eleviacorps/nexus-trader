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
    _default_windows,
    _enrich_signals,
    _load_feature_frame,
)
from src.v18.kimi_system_prompt import build_kimi_user_message
from src.v24_4_2.recovery_runtime import build_validation_result, write_validation_outputs
from src.v24_4_2.threshold_optimizer import ThresholdConfig

OUTPUT_DIR = OUTPUTS_DIR / "v24_4_2"


def _load_best_config() -> ThresholdConfig:
    best_path = OUTPUT_DIR / "best_threshold_config.json"
    if best_path.exists():
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        return ThresholdConfig(
            trend_up=float(config.get("trend_up", 0.54)),
            trend_down=float(config.get("trend_down", 0.64)),
            breakout=float(config.get("breakout", 0.58)),
            range_value=float(config.get("range_value", 0.60)),
            cooldown_decay=float(config.get("cooldown_decay", 0.75)),
            cluster_radius=float(config.get("cluster_radius", 0.25)),
            size_multiplier=float(config.get("size_multiplier", 1.00)),
        )
    return ThresholdConfig(
        trend_up=0.54,
        trend_down=0.64,
        breakout=0.58,
        range_value=0.60,
        cooldown_decay=0.75,
        cluster_radius=0.25,
        size_multiplier=1.00,
    )


def _build_windows_data() -> list[tuple[Any, pd.DataFrame, pd.DataFrame]]:
    windows_data: list[tuple[Any, pd.DataFrame, pd.DataFrame]] = []
    for window in _default_windows():
        frame = _load_feature_frame(window.start, window.end, prelude_days=45)
        signals = _build_or_load_signals(window, frame)
        candidates = _enrich_signals(signals, frame)
        if candidates.empty:
            continue
        windows_data.append((window, candidates, frame))
    return windows_data


def _kimi_key_coverage_report() -> dict[str, Any]:
    context = {
        "market": {"current_price": 2350.25},
        "simulation": {
            "scenario_bias": "bullish",
            "overall_confidence": 0.64,
            "cabr_score": 0.61,
            "cpm_score": 0.72,
            "cone_width_pips": 142.0,
            "hurst_overall": 0.58,
            "hurst_asymmetry": 0.11,
            "testosterone_index": {"retail": 0.32, "institutional": 0.18},
        },
        "technical_analysis": {"structure": "bullish", "location": "discount", "rsi_14": 57.0, "atr_14": 18.2, "equilibrium": 2346.8},
        "bot_swarm": {"aggregate": {"signal": "bullish", "bullish_probability": 0.68, "disagreement": 0.12}},
        "sqt": {"label": "HOT", "rolling_accuracy": 0.7},
        "live_performance": {"rolling_win_rate_10": 0.6, "consecutive_losses": 1, "equity": 1000.0},
        "v21_runtime": {
            "runtime_version": "v21_local",
            "v21_dir_15m_prob": 0.62,
            "v21_bimamba_prob": 0.59,
            "v21_ensemble_prob": 0.64,
            "v21_meta_label_prob": 0.58,
            "v21_disagree_prob": 0.12,
            "v21_dangerous_branch_count": 1,
        },
        "v22_runtime": {
            "online_hmm": {"regime_confidence": 0.72, "persistence_count": 4},
            "circuit_breaker": {"state": "CLEAR", "trading_allowed": True},
            "ensemble": {"agreement_rate": 0.8, "meta_label_prob": 0.61, "risk_score": 0.21},
            "risk_check": {"rr_ratio": 1.8},
        },
    }
    message = build_kimi_user_message(context, "XAUUSD")
    expected_markers = [
        "V21 LOCAL RUNTIME",
        "V22 RISK STACK",
        "xLSTM 15m probability",
        "BiMamba probability",
        "Ensemble probability",
        "Meta-label probability",
        "Online HMM confidence",
        "Circuit breaker state",
        "Ensemble agreement rate",
        "Meta-label accept probability",
    ]
    covered = [marker for marker in expected_markers if marker in message]
    return {
        "expected_markers": expected_markers,
        "covered_markers": covered,
        "coverage_ratio": float(len(covered) / max(len(expected_markers), 1)),
        "model_policy_note": "V25 Claude gateway model order is enforced independently in src/service/claude_trade_gateway.py",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = _load_best_config()
    windows_data = _build_windows_data()
    if not windows_data:
        raise RuntimeError("No validation windows available for v24_4_2.")

    result = build_validation_result(windows_data, config)
    output_paths = write_validation_outputs(result, output_dir=OUTPUT_DIR)

    coverage = _kimi_key_coverage_report()
    coverage_path = OUTPUT_DIR / "kimi_key_coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "final_validation_json": str(output_paths["json"]),
                "final_validation_md": str(output_paths["md"]),
                "kimi_key_coverage": str(coverage_path),
                "aggregate_metrics": result.aggregate_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

