from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_backtrader_week import _build_signal_frame


def _commission(notional: float, commission_bps: float) -> float:
    return float(notional) * (float(commission_bps) / 10000.0)


def _entry_price(raw_open: float, direction: int, slippage_perc: float) -> float:
    if direction > 0:
        return float(raw_open) * (1.0 + float(slippage_perc))
    return float(raw_open) * (1.0 - float(slippage_perc))


def _exit_price(raw_close: float, direction: int, slippage_perc: float) -> float:
    if direction > 0:
        return float(raw_close) * (1.0 - float(slippage_perc))
    return float(raw_close) * (1.0 + float(slippage_perc))


def run_simple_backtest(
    frame: pd.DataFrame,
    *,
    capital: float,
    stake: float,
    hold_bars: int,
    commission_bps: float,
    slippage_perc: float,
) -> dict[str, Any]:
    rows = frame.reset_index().copy()
    cash = float(capital)
    pending_signal = 0
    position: dict[str, Any] | None = None
    wins = 0
    losses = 0
    gross_pnl_sum = 0.0
    net_pnl_sum = 0.0
    trades: list[dict[str, Any]] = []

    for idx, row in rows.iterrows():
        current_signal = int(row["signal"])

        if pending_signal != 0 and position is None:
            direction = int(pending_signal)
            qty = float(stake)
            open_price = float(row["open"])
            fill_price = _entry_price(open_price, direction, slippage_perc)
            entry_notional = abs(qty * fill_price)
            entry_fee = _commission(entry_notional, commission_bps)
            cash -= entry_fee
            position = {
                "direction": direction,
                "qty": qty,
                "entry_index": int(idx),
                "entry_time": str(row["datetime"]),
                "entry_price": fill_price,
                "entry_fee": entry_fee,
            }
            pending_signal = 0

        if position is not None:
            bars_held = int(idx) - int(position["entry_index"])
            if bars_held >= int(hold_bars):
                direction = int(position["direction"])
                qty = float(position["qty"])
                raw_close = float(row["close"])
                fill_price = _exit_price(raw_close, direction, slippage_perc)
                exit_notional = abs(qty * fill_price)
                exit_fee = _commission(exit_notional, commission_bps)
                gross_pnl = float(direction) * qty * (fill_price - float(position["entry_price"]))
                net_pnl = gross_pnl - float(position["entry_fee"]) - exit_fee
                cash += gross_pnl - exit_fee
                gross_pnl_sum += gross_pnl
                net_pnl_sum += net_pnl
                if net_pnl > 0:
                    wins += 1
                elif net_pnl < 0:
                    losses += 1
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": str(row["datetime"]),
                        "direction": direction,
                        "qty": qty,
                        "entry_price": round(float(position["entry_price"]), 6),
                        "exit_price": round(fill_price, 6),
                        "gross_pnl": round(gross_pnl, 6),
                        "net_pnl": round(net_pnl, 6),
                        "bars_held": bars_held,
                    }
                )
                position = None

        if position is None and pending_signal == 0 and current_signal != 0 and idx < len(rows) - 1:
            pending_signal = current_signal

    trade_count = len(trades)
    report = {
        "bars": int(len(rows)),
        "signal_count": int((rows["signal"] != 0).sum()),
        "trade_count": int(trade_count),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(float(wins / trade_count), 6) if trade_count else 0.0,
        "gross_pnl_sum": round(float(gross_pnl_sum), 6),
        "net_pnl_sum": round(float(net_pnl_sum), 6),
        "avg_net_pnl": round(float(net_pnl_sum / trade_count), 6) if trade_count else 0.0,
        "start_cash": round(float(capital), 6),
        "final_cash": round(float(cash), 6),
        "return_pct": round(float((cash / capital - 1.0) * 100.0), 6) if capital else 0.0,
        "stake": float(stake),
        "hold_bars": int(hold_bars),
        "commission_bps": float(commission_bps),
        "slippage_perc": float(slippage_perc),
        "trades": trades,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a simple transparent one-week local execution backtest from V8 signals.")
    parser.add_argument("--run-tag", default="mh12_full_v8")
    parser.add_argument("--horizon", default="15m")
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--stake", type=float, default=0.01)
    parser.add_argument("--hold-bars", type=int, default=15)
    parser.add_argument("--commission-bps", type=float, default=2.0)
    parser.add_argument("--slippage-perc", type=float, default=0.0002)
    parser.add_argument("--disable-external-gate", action="store_true")
    parser.add_argument("--gate-threshold-override", type=float, default=None)
    args = parser.parse_args()

    frame, meta = _build_signal_frame(
        args.run_tag,
        args.horizon,
        use_external_gate=not args.disable_external_gate,
        gate_threshold_override=args.gate_threshold_override,
    )
    report = run_simple_backtest(
        frame,
        capital=float(args.capital),
        stake=float(args.stake),
        hold_bars=int(args.hold_bars),
        commission_bps=float(args.commission_bps),
        slippage_perc=float(args.slippage_perc),
    )
    report["run_tag"] = args.run_tag
    report["horizon"] = args.horizon
    report["start"] = str(frame.index.min())
    report["end"] = str(frame.index.max())
    report["meta"] = meta

    output_path = PROJECT_ROOT / "outputs" / "evaluation" / f"simple_week_{args.run_tag}_{args.horizon}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)
    print(json.dumps({k: v for k, v in report.items() if k != "trades"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
