from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.v14.rsc import RegimeStratifiedCalibrator


def _iter_trade_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    trade_list: list[dict[str, Any]] = []
    if isinstance(report.get("months"), list):
        for month in report["months"]:
            trade_list.extend(month.get("trade_log", []))
            trade_list.extend(month.get("trades", []))
    if isinstance(report.get("monthly_results"), list):
        for month in report["monthly_results"]:
            trade_list.extend(month.get("trade_log", []))
            trade_list.extend(month.get("trades", []))
    if isinstance(report.get("trades"), list):
        trade_list.extend(report["trades"])
    if isinstance(report.get("trade_log"), list):
        trade_list.extend(report["trade_log"])
    return trade_list


def _derive_outcome(trade: dict[str, Any]) -> bool | None:
    outcome = str(trade.get("outcome", "")).strip().lower()
    if outcome == "win":
        return True
    if outcome == "loss":
        return False
    for key in ("net_pnl_usd", "net_pnl", "gross_pnl", "pnl_pips"):
        value = trade.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            return True
        if numeric < 0.0:
            return False
    return None


def bootstrap_from_walkforward(
    walkforward_paths: list[str | Path],
    rsc: RegimeStratifiedCalibrator,
) -> tuple[RegimeStratifiedCalibrator, int]:
    total = 0
    for raw_path in walkforward_paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"[CBWF] Skipping {path} - not found", flush=True)
            continue

        report = json.loads(path.read_text(encoding="utf-8"))
        for trade in _iter_trade_records(report):
            won = _derive_outcome(trade)
            if won is None:
                continue
            regime = str(trade.get("regime", trade.get("dominant_regime", "unknown")))
            score = float(trade.get("uts_score", trade.get("cabr_score", 0.5)))
            rsc.record_outcome(score, regime, won)
            total += 1

    print(f"[CBWF] Bootstrapped {total} trades across {len(walkforward_paths)} walk-forward reports", flush=True)
    print(f"[CBWF] Per-regime counts: {dict(rsc._counts)}", flush=True)
    return rsc, total


def build_bootstrapped_rsc(
    walkforward_paths: list[str | Path],
    save_path: str | Path,
) -> RegimeStratifiedCalibrator:
    rsc = RegimeStratifiedCalibrator()
    rsc, total = bootstrap_from_walkforward(walkforward_paths, rsc)
    save_path = Path(save_path)
    rsc.save(save_path)
    print(f"[CBWF] Saved bootstrapped RSC with {total} examples to {save_path}", flush=True)
    summary = rsc.summary()
    print(f"[CBWF] Summary: {json.dumps(summary, indent=2)}", flush=True)
    return rsc
