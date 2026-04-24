"""V27.1 Backtest Engine with Execution Policy and Risk Sizing."""

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.v27.execution_policy import (
    ExecutionPolicy,
    HoldTarget,
    get_hold_target,
    check_early_exit,
    compute_realized_rr,
)
from src.v27.risk_manager import create_risk_manager
from src.v27.short_horizon_predictor import create_short_horizon_predictor
from src.v26.diffusion.regime_generator import RegimeGeneratorConfig, RegimeDiffusionPathGenerator


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: str
    direction: str
    confidence: float
    rr_target: float
    hold_target_min: float
    hold_target_max: float
    entry: float
    exit: float
    exit_reason: str
    realized_hold_min: float
    realized_rr: float
    lot_size: float
    pnl_dollars: float
    account_balance: float


def run_backtest(
    num_trades: int = 100,
    account_balance: float = 1000.0,
    initial_price: float = 100.0,
    num_futures: int = 16,
    steps: int = 10,
) -> List[TradeRecord]:
    """Run backtest with V27.1 execution policy."""

    # Load generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RegimeGeneratorConfig(
        in_channels=144,
        sequence_length=120,
        temporal_gru_dim=256,
        temporal_layers=2,
        context_len=256,
        num_regimes=9,
        regime_embed_dim=16,
        temporal_film_dim=272,
    )
    base_gen = RegimeDiffusionPathGenerator(config=config, device=str(device))

    ckpt_path = "models/v26/diffusion_phase1_final.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = base_gen.model.state_dict()
        compatible = {k: v for k, v in ckpt.get("ema", ckpt.get("model", {})).items()
                     if k in model_state and model_state[k].shape == v.shape}
        base_gen.model.load_state_dict(compatible, strict=False)

    predictor = create_short_horizon_predictor(base_gen, device=str(device))
    predictor.num_futures = num_futures
    predictor.confidence_threshold = 0.60
    predictor.validity_minutes = 15

    # Execution
    policy = ExecutionPolicy()
    risk_manager = create_risk_manager(account_balance, stage=1)

    trades = []
    balance = account_balance

    start_time = datetime(2024, 1, 1, 9, 30)

    for i in range(num_trades):
        timestamp = start_time + timedelta(minutes=i * 30)

        # Generate prediction
        regime_probs = torch.rand(9)
        regime_probs = regime_probs / regime_probs.sum()
        past_context = torch.randn(256, 144)

        result = predictor.predict_15min_trade(
            past_context=past_context,
            regime_probs=regime_probs,
            current_price=initial_price + random.uniform(-2, 2),
            steps=steps,
        )

        if result.decision == "HOLD":
            continue

        # Get hold target and RR
        hold_target = get_hold_target(result.confidence)

        # Calculate TP/SL based on RR
        if result.decision == "BUY":
            entry = result.entry_price
            stop_distance = abs(result.take_profit - entry) / hold_target.rr_target
            sl = entry - stop_distance
            tp = entry + stop_distance * hold_target.rr_target
        else:
            entry = result.entry_price
            stop_distance = abs(entry - result.take_profit) / hold_target.rr_target
            sl = entry + stop_distance
            tp = entry - stop_distance * hold_target.rr_target

        # Calculate lot size
        lot_size = risk_manager.compute_lot_size(entry, sl)

        # Simulate price movement (simplified random walk)
        future_price = initial_price + random.uniform(-5, 5)
        expiry_time = timestamp.timestamp() + 15 * 60

        # Simulate price path
        current_price = entry
        trade_exit_reason = None
        exit_price = entry
        exit_time = expiry_time

        for minute in range(16):
            current_time = timestamp.timestamp() + minute * 60
            current_price = entry + random.uniform(-0.5, 0.5)

            # Check TP/SL
            if result.decision == "BUY":
                if current_price >= tp:
                    trade_exit_reason = "tp_hit"
                    exit_price = tp
                    exit_time = current_time
                    break
                if current_price <= sl:
                    trade_exit_reason = "sl_hit"
                    exit_price = sl
                    exit_time = current_time
                    break
            else:
                if current_price <= tp:
                    trade_exit_reason = "tp_hit"
                    exit_price = tp
                    exit_time = current_time
                    break
                if current_price >= sl:
                    trade_exit_reason = "sl_hit"
                    exit_price = sl
                    exit_time = current_time
                    break

        if trade_exit_reason is None:
            # Expiry
            trade_exit_reason = "expiry"
            exit_price = current_price
            exit_time = expiry_time

        # Calculate P&L
        realized_rr = compute_realized_rr(entry, exit_price, sl, result.decision)
        pnl = realized_rr * risk_manager.get_risk_dollars()

        balance += pnl

        if pnl > 0:
            risk_manager.on_win()
        else:
            risk_manager.on_loss()

        realized_hold_min = (exit_time - timestamp.timestamp()) / 60.0

        trade = TradeRecord(
            timestamp=timestamp.isoformat(),
            direction=result.decision,
            confidence=result.confidence,
            rr_target=hold_target.rr_target,
            hold_target_min=hold_target.min_min,
            hold_target_max=hold_target.max_min,
            entry=entry,
            exit=exit_price,
            exit_reason=trade_exit_reason,
            realized_hold_min=realized_hold_min,
            realized_rr=realized_rr,
            lot_size=lot_size,
            pnl_dollars=pnl,
            account_balance=balance,
        )
        trades.append(trade)

        initial_price = current_price

    return trades


def analyze_trades(trades: List[TradeRecord]) -> dict:
    """Analyze trade results."""
    if not trades:
        return {"error": "No trades"}

    directions = [t.direction for t in trades]
    exit_reasons = [t.exit_reason for t in trades]
    confidences = [t.confidence for t in trades]
    hold_times = [t.realized_hold_min for t in trades]
    rrs = [t.realized_rr for t in trades]
    pnls = [t.pnl_dollars for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total = len(trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total if total > 0 else 0

    avg_rr = np.mean(rrs) if rrs else 0
    avg_hold = np.mean(hold_times) if hold_times else 0
    avg_conf = np.mean(confidences) if confidences else 0

    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

    avg_pnl = np.mean(pnls)
    total_pnl = sum(pnls)

    ending_balance = trades[-1].account_balance if trades else 1000
    initial_balance = trades[0].account_balance - total_pnl
    return_pct = (ending_balance - initial_balance) / initial_balance * 100 if initial_balance > 0 else 0

    # Confidence buckets
    bucket_60_67 = [t for t in trades if 0.60 <= t.confidence < 0.67]
    bucket_67_74 = [t for t in trades if 0.67 <= t.confidence < 0.74]
    bucket_74_82 = [t for t in trades if 0.74 <= t.confidence < 0.82]
    bucket_82_plus = [t for t in trades if t.confidence >= 0.82]

    def bucket_stats(bucket):
        if not bucket:
            return {"count": 0, "win_rate": 0, "avg_rr": 0, "avg_hold": 0}
        wins_b = sum(1 for t in bucket if t.pnl_dollars > 0)
        return {
            "count": len(bucket),
            "win_rate": wins_b / len(bucket),
            "avg_rr": np.mean([t.realized_rr for t in bucket]),
            "avg_hold": np.mean([t.realized_hold_min for t in bucket]),
        }

    bucket_report = {
        "0.60-0.67": bucket_stats(bucket_60_67),
        "0.67-0.74": bucket_stats(bucket_67_74),
        "0.74-0.82": bucket_stats(bucket_74_82),
        "0.82+": bucket_stats(bucket_82_plus),
    }

    # Exit reasons
    expiry_pct = exit_reasons.count("expiry") / total * 100 if total > 0 else 0
    tp_pct = exit_reasons.count("tp_hit") / total * 100 if total > 0 else 0
    sl_pct = exit_reasons.count("sl_hit") / total * 100 if total > 0 else 0

    return {
        "total_trades": total,
        "buy_count": directions.count("BUY"),
        "sell_count": directions.count("SELL"),
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "avg_hold_min": avg_hold,
        "avg_confidence": avg_conf,
        "profit_factor": profit_factor,
        "expectancy": avg_pnl,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "ending_balance": ending_balance,
        "expiry_pct": expiry_pct,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "confidence_buckets": bucket_report,
    }


def main():
    print("=" * 60)
    print("V27.1 Backtest with Execution Policy")
    print("=" * 60)

    trades = run_backtest(num_trades=100, account_balance=1000.0)

    # Save trade log
    trade_log = []
    for t in trades:
        trade_log.append({
            "timestamp": t.timestamp,
            "direction": t.direction,
            "confidence": t.confidence,
            "rr_target": t.rr_target,
            "hold_target_min": t.hold_target_min,
            "hold_target_max": t.hold_target_max,
            "entry": t.entry,
            "exit": t.exit,
            "exit_reason": t.exit_reason,
            "realized_hold_min": t.realized_hold_min,
            "realized_rr": t.realized_rr,
            "lot_size": t.lot_size,
            "pnl_dollars": t.pnl_dollars,
            "account_balance": t.account_balance,
        })

    Path("outputs/v27_1").mkdir(parents=True, exist_ok=True)
    with open("outputs/v27_1/trade_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trade_log[0].keys())
        writer.writeheader()
        writer.writerows(trade_log)

    # Analyze
    metrics = analyze_trades(trades)

    # Save summary
    with open("outputs/v27_1/backtest_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save bucket report
    with open("outputs/v27_1/confidence_bucket_report.json", "w") as f:
        json.dump(metrics.get("confidence_buckets", {}), f, indent=2)

    # Equity curve
    equity = [{"balance": t.account_balance, "timestamp": t.timestamp} for t in trades]
    with open("outputs/v27_1/equity_curve.csv", "w") as f:
        f.write("timestamp,balance\n")
        for e in equity:
            f.write(f"{e['timestamp']},{e['balance']}\n")

    # Print results
    print("\nRESULTS")
    print("=" * 40)
    print(f"Total trades: {metrics['total_trades']}")
    print(f"BUY: {metrics['buy_count']}, SELL: {metrics['sell_count']}")
    print(f"Win rate: {metrics['win_rate']*100:.1f}%")
    print(f"Avg hold time: {metrics['avg_hold_min']:.1f} min")
    print(f"Avg RR: {metrics['avg_rr']:.2f}")
    print(f"Avg confidence: {metrics['avg_confidence']:.3f}")
    print(f"Return: {metrics['return_pct']:.1f}%")
    print(f"Expiry %: {metrics['expiry_pct']:.1f}%")
    print(f"TP %: {metrics['tp_pct']:.1f}%")
    print(f"SL %: {metrics['sl_pct']:.1f}%")

    # Check success criteria
    success = (
        metrics['avg_hold_min'] >= 3.0 and
        metrics['avg_hold_min'] <= 10.0 and
        metrics['win_rate'] >= 0.50 and
        metrics['return_pct'] > 0 and
        metrics['expiry_pct'] < 50
    )

    print(f"\n{'SUCCESS' if success else 'NEEDS IMPROVEMENT'}")

    # Save report
    report = f"""# V27.1 Execution Report

## Summary
- Total trades: {metrics['total_trades']}
- Win rate: {metrics['win_rate']*100:.1f}%
- Average hold time: {metrics['avg_hold_min']:.1f} min
- Average RR: {metrics['avg_rr']:.2f}
- Return: {metrics['return_pct']:.1f}%

## Exit Analysis
- Expiry: {metrics['expiry_pct']:.1f}%
- TP hit: {metrics['tp_pct']:.1f}%
- SL hit: {metrics['sl_pct']:.1f}%

## Confidence Buckets
{json.dumps(metrics.get('confidence_buckets', {}), indent=2)}

## Success Criteria
- Hold time 3-10 min: {'PASS' if 3 <= metrics['avg_hold_min'] <= 10 else 'FAIL'}
- Win rate >= 50%: {'PASS' if metrics['win_rate'] >= 0.50 else 'FAIL'}
- Return > 0%: {'PASS' if metrics['return_pct'] > 0 else 'FAIL'}
- Expiry < 50%: {'PASS' if metrics['expiry_pct'] < 50 else 'FAIL'}

## Recommendation
{'Proceed to live paper trading' if success else 'Continue optimization'}
"""

    with open("outputs/v27_1/V27_1_execution_report.md", "w") as f:
        f.write(report)

    print(f"\nReport saved to: outputs/v27_1/")


if __name__ == "__main__":
    main()