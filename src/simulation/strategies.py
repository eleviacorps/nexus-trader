from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping


@dataclass(frozen=True)
class StrategySignal:
    direction: int
    confidence: float


class Strategies:
    """Calibrated 1-minute strategy heuristics derived from the legacy notebooks."""

    @staticmethod
    def trend_following(row: Mapping[str, float]) -> StrategySignal:
        ema_cross = float(row.get("ema_cross", 0.0))
        rsi = float(row.get("rsi_14", 50.0))
        macd_hist = float(row.get("macd_hist", 0.0))

        if ema_cross > 0 and macd_hist > 0 and 55 < rsi < 75:
            return StrategySignal(direction=-1, confidence=min(0.80, 0.48 + abs(macd_hist) / 25 + (rsi - 55) / 120))
        if ema_cross < 0 and macd_hist < 0 and 25 < rsi < 45:
            return StrategySignal(direction=1, confidence=min(0.80, 0.48 + abs(macd_hist) / 25 + (45 - rsi) / 120))
        return StrategySignal(direction=0, confidence=0.0)

    @staticmethod
    def mean_reversion(row: Mapping[str, float]) -> StrategySignal:
        bb_pct = float(row.get("bb_pct", 0.5))
        rsi = float(row.get("rsi_14", 50.0))
        if bb_pct < 0.25 and rsi < 42:
            return StrategySignal(direction=1, confidence=min(0.90, 0.50 + (42 - rsi) / 42 * 0.4))
        if bb_pct > 0.75 and rsi > 58:
            return StrategySignal(direction=-1, confidence=min(0.90, 0.50 + (rsi - 58) / 42 * 0.4))
        return StrategySignal(direction=0, confidence=0.0)

    @staticmethod
    def ict_liquidity_hunt(row: Mapping[str, float]) -> StrategySignal:
        dist_high = float(row.get("dist_to_high", 999.0))
        dist_low = float(row.get("dist_to_low", 999.0))
        rsi = float(row.get("rsi_14", 50.0))

        if 0 < dist_high < 1.0 and rsi > 58:
            return StrategySignal(direction=-1, confidence=min(0.85, 0.53 + (1.0 - dist_high) * 0.32))
        if 0 < dist_low < 1.0 and rsi < 42:
            return StrategySignal(direction=1, confidence=min(0.85, 0.53 + (1.0 - dist_low) * 0.32))
        return StrategySignal(direction=0, confidence=0.0)

    @staticmethod
    def smc_structure(row: Mapping[str, float]) -> StrategySignal:
        hh = float(row.get("hh", 0.0))
        ll = float(row.get("ll", 0.0))
        ema_cross = float(row.get("ema_cross", 0.0))
        rsi = float(row.get("rsi_14", 50.0))

        if hh and ema_cross > 0 and 58 < rsi < 75:
            return StrategySignal(direction=-1, confidence=0.60 + min(0.20, (rsi - 58) / 85))
        if ll and ema_cross < 0 and 25 < rsi < 42:
            return StrategySignal(direction=1, confidence=0.60 + min(0.20, (42 - rsi) / 85))
        return StrategySignal(direction=0, confidence=0.0)

    @staticmethod
    def momentum_scalp(row: Mapping[str, float]) -> StrategySignal:
        rsi_7 = float(row.get("rsi_7", 50.0))
        macd_hist = float(row.get("macd_hist", 0.0))
        body_pct = float(row.get("body_pct", 0.5))

        if rsi_7 > 62 and macd_hist > 0 and body_pct > 0.5:
            return StrategySignal(direction=-1, confidence=0.55 + min(0.25, (rsi_7 - 62) / 40))
        if rsi_7 < 38 and macd_hist < 0 and body_pct > 0.5:
            return StrategySignal(direction=1, confidence=0.55 + min(0.25, (38 - rsi_7) / 40))
        return StrategySignal(direction=0, confidence=0.0)


STRATEGY_ORDER = [
    "trend",
    "mean_rev",
    "ict",
    "smc",
    "momentum",
]


def strategy_map() -> Dict[str, Callable[[Mapping[str, float]], StrategySignal]]:
    return {
        "trend": Strategies.trend_following,
        "mean_rev": Strategies.mean_reversion,
        "ict": Strategies.ict_liquidity_hunt,
        "smc": Strategies.smc_structure,
        "momentum": Strategies.momentum_scalp,
    }


def evaluate_strategy_accuracy(
    rows: Iterable[Mapping[str, float]],
    targets: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    rows_list = list(rows)
    targets_list = list(targets)
    if len(rows_list) != len(targets_list):
        raise ValueError("rows and targets must have the same length")

    for name, fn in strategy_map().items():
        active = 0
        correct = 0
        for row, target in zip(rows_list, targets_list):
            signal = fn(row)
            if signal.direction == 0:
                continue
            active += 1
            target_direction = 1 if int(target) > 0 else -1
            if signal.direction == target_direction:
                correct += 1
        metrics[name] = {
            "active_rate": active / max(1, len(rows_list)),
            "accuracy": correct / max(1, active),
            "active_count": float(active),
        }
    return metrics


def assert_strategy_floor(metrics: Mapping[str, Mapping[str, float]], threshold: float = 0.52) -> None:
    failures = [name for name, values in metrics.items() if float(values.get("accuracy", 0.0)) < threshold]
    if failures:
        joined = ", ".join(sorted(failures))
        raise AssertionError(f"Strategies below {threshold:.0%} accuracy: {joined}")
