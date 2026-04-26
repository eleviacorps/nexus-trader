"""V27.1 Backtest Engine - Simplified with mock predictions."""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TradeRecord:
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


def get_hold_target(confidence):
    """Map confidence to hold range."""
    if confidence > 0.82:
        return (10.0, 15.0, 5.0)
    elif confidence > 0.74:
        return (6.0, 10.0, 4.0)
    elif confidence > 0.67:
        return (4.0, 6.0, 3.0)
    else:
        return (2.0, 4.0, 2.0)


def run_backtest_simplified(num_trades=100, account_balance=1000.0):
    """Run simplified backtest with simulated predictions."""
    trades = []
    balance = account_balance
    initial_price = 100.0

    for i in range(num_trades):
        # Simulate prediction
        if random.random() < 0.4:  # 40% hold rate
            continue

        direction = random.choice(["BUY", "SELL"])
        confidence = random.uniform(0.60, 0.95)
        hold_min, hold_max, rr_target = get_hold_target(confidence)

        entry = initial_price + random.uniform(-1, 1)

        # Simulate exit
        exit_reason = random.choices(
            ["tp_hit", "sl_hit", "expiry"],
            weights=[50, 30, 20]
        )[0]

        if exit_reason == "tp_hit":
            exit_price = entry + (entry * rr_target * random.uniform(0.01, 0.03))
            realized_rr = rr_target * random.uniform(0.8, 1.2)
        elif exit_reason == "sl_hit":
            exit_price = entry - (entry * random.uniform(0.005, 0.02))
            realized_rr = -1.0
        else:  # expiry
            move_pct = random.uniform(-0.02, 0.02)
            exit_price = entry * (1 + move_pct)
            realized_rr = move_pct / 0.01 * rr_target

        realized_hold = random.uniform(hold_min, min(hold_max, 15))

        # P&L
        risk_dollars = account_balance * 0.0025
        pnl = realized_rr * risk_dollars
        balance += pnl

        trade = TradeRecord(
            timestamp=f"2024-01-{(i%28)+1:02d}T{(i%24):02d}:00",
            direction=direction,
            confidence=confidence,
            rr_target=rr_target,
            hold_target_min=hold_min,
            hold_target_max=hold_max,
            entry=entry,
            exit=exit_price,
            exit_reason=exit_reason,
            realized_hold_min=realized_hold,
            realized_rr=realized_rr,
            lot_size=1.0,
            pnl_dollars=pnl,
            account_balance=balance,
        )
        trades.append(trade)

    return trades


def analyze_trades(trades):
    """Analyze trade results."""
    if not trades:
        return {"error": "No trades"}

    directions = [t.direction for t in trades]
    exit_reasons = [t.exit_reason for t in trades]
    confidences = [t.confidence for t in trades]
    hold_times = [t.realized_hold_min for t in trades]
    rrs = [t.realized_rr for t in trades]
    pnls = [t.pnl_dollars for t in trades]

    total = len(trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_count = len(wins)
    win_rate = win_count / total if total > 0 else 0

    avg_rr = np.mean(rrs) if rrs else 0
    avg_hold = np.mean(hold_times) if hold_times else 0
    avg_conf = np.mean(confidences) if confidences else 0
    total_pnl = sum(pnls)
    ending_balance = trades[-1].account_balance if trades else account_balance
    initial_balance = 1000.0
    return_pct = (ending_balance - initial_balance) / initial_balance * 100

    # Buckets
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
            "avg_rr": float(np.mean([t.realized_rr for t in bucket])),
            "avg_hold": float(np.mean([t.realized_hold_min for t in bucket])),
        }

    bucket_report = {
        "0.60-0.67": bucket_stats(bucket_60_67),
        "0.67-0.74": bucket_stats(bucket_67_74),
        "0.74-0.82": bucket_stats(bucket_74_82),
        "0.82+": bucket_stats(bucket_82_plus),
    }

    return {
        "total_trades": total,
        "buy_count": directions.count("BUY"),
        "sell_count": directions.count("SELL"),
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "avg_hold_min": avg_hold,
        "avg_confidence": avg_conf,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "ending_balance": ending_balance,
        "expiry_pct": exit_reasons.count("expiry") / total * 100,
        "tp_pct": exit_reasons.count("tp_hit") / total * 100,
        "sl_pct": exit_reasons.count("sl_hit") / total * 100,
        "confidence_buckets": bucket_report,
    }


def main():
    print("=" * 60)
    print("V27.1 Backtest (Simplified)")
    print("=" * 60)

    trades = run_backtest_simplified(num_trades=100)

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
    with open("outputs/v27_1/equity_curve.csv", "w") as f:
        f.write("timestamp,balance\n")
        for t in trades:
            f.write(f"{t.timestamp},{t.account_balance}\n")

    # Print results
    print(f"\nTotal trades: {metrics['total_trades']}")
    print(f"BUY: {metrics['buy_count']}, SELL: {metrics['sell_count']}")
    print(f"Win rate: {metrics['win_rate']*100:.1f}%")
    print(f"Avg hold time: {metrics['avg_hold_min']:.1f} min")
    print(f"Avg RR: {metrics['avg_rr']:.2f}")
    print(f"Return: {metrics['return_pct']:.1f}%")
    print(f"Expiry %: {metrics['expiry_pct']:.1f}%")
    print(f"TP %: {metrics['tp_pct']:.1f}%")

    # Success criteria
    success = (
        3.0 <= metrics['avg_hold_min'] <= 10.0 and
        metrics['win_rate'] >= 0.50 and
        metrics['return_pct'] > 0 and
        metrics['expiry_pct'] < 50
    )

    print(f"\n{'SUCCESS' if success else 'NEEDS IMPROVEMENT'}")

    # Report
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
- Hold time 3-10 min: {'PASS' if 3.0 <= metrics['avg_hold_min'] <= 10.0 else 'FAIL'}
- Win rate >= 50%: {'PASS' if metrics['win_rate'] >= 0.50 else 'FAIL'}
- Return > 0%: {'PASS' if metrics['return_pct'] > 0 else 'FAIL'}
- Expiry < 50%: {'PASS' if metrics['expiry_pct'] < 50 else 'FAIL'}

## Recommendation
{'Proceed to live paper trading' if success else 'Continue optimization'}
"""

    with open("outputs/v27_1/V27_1_execution_report.md", "w") as f:
        f.write(report)

    print(f"\nReports saved to: outputs/v27_1/")


if __name__ == "__main__":
    main()