"""V27.1 Execution Policy for 15-Minute Trade Predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HoldTarget:
    """Target hold range based on confidence."""
    min_min: float
    max_min: float
    rr_target: float


def get_hold_target(confidence: float) -> HoldTarget:
    """Map confidence to desired hold range and RR target.

    Args:
        confidence: Prediction confidence (0.0 - 1.0)

    Returns:
        HoldTarget with min/max hold times and RR ratio
    """
    if confidence > 0.82:
        return HoldTarget(min_min=10.0, max_min=15.0, rr_target=5.0)
    elif confidence > 0.74:
        return HoldTarget(min_min=6.0, max_min=10.0, rr_target=4.0)
    elif confidence > 0.67:
        return HoldTarget(min_min=4.0, max_min=6.0, rr_target=3.0)
    else:
        return HoldTarget(min_min=2.0, max_min=4.0, rr_target=2.0)


@dataclass
class ExecutionPolicy:
    """Execution policy for V27.1 trades."""

    min_hold_min: float = 2.5
    confidence_collapse_threshold: float = 0.45
    validity_minutes: int = 15


@dataclass
class TradeExit:
    """Result of trade exit."""
    reason: str  # tp_hit, sl_hit, expiry, confidence_collapse, opposite_signal
    exit_price: float
    realized_hold_min: float
    realized_rr: float
    pnl_dollars: float


def check_early_exit(
    policy: ExecutionPolicy,
    trade_start_time: float,
    current_time: float,
    confidence: float,
    current_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    hold_target: HoldTarget,
    opposite_signal: bool = False,
) -> Optional[TradeExit]:
    """Check if trade should exit early.

    Args:
        policy: Execution policy
        trade_start_time: When trade was entered
        current_time: Current timestamp
        confidence: Current prediction confidence
        current_price: Current market price
        entry_price: Entry price
        stop_loss: Stop loss level
        take_profit: Take profit level
        direction: BUY or SELL
        hold_target: Target hold range
        opposite_signal: Whether opposite signal appeared

    Returns:
        TradeExit if should exit, None otherwise
    """
    elapsed_min = (current_time - trade_start_time) / 60.0

    # Cannot exit before min hold time unless TP/SL/explicit signal
    if elapsed_min < policy.min_hold_min:
        tp_triggered = False
        sl_triggered = False

        if direction == "BUY":
            tp_triggered = current_price >= take_profit
            sl_triggered = current_price <= stop_loss
        else:
            tp_triggered = current_price <= take_profit
            sl_triggered = current_price >= stop_loss

        if sl_triggered:
            return TradeExit(
                reason="sl_hit",
                exit_price=current_price,
                realized_hold_min=elapsed_min,
                realized_rr=-1.0,
                pnl_dollars=0.0,
            )
        if tp_triggered:
            return TradeExit(
                reason="tp_hit",
                exit_price=current_price,
                realized_hold_min=elapsed_min,
                realized_rr=hold_target.rr_target,
                pnl_dollars=0.0,
            )
        return None

    # After min hold, check other conditions
    # Check TP/SL
    tp_triggered = sl_triggered = False
    if direction == "BUY":
        tp_triggered = current_price >= take_profit
        sl_triggered = current_price <= stop_loss
    else:
        tp_triggered = current_price <= take_profit
        sl_triggered = current_price >= stop_loss

    if sl_triggered:
        return TradeExit(
            reason="sl_hit",
            exit_price=current_price,
            realized_hold_min=elapsed_min,
            realized_rr=-1.0,
            pnl_dollars=0.0,
        )
    if tp_triggered:
        return TradeExit(
            reason="tp_hit",
            exit_price=current_price,
            realized_hold_min=elapsed_min,
            realized_rr=hold_target.rr_target,
            pnl_dollars=0.0,
        )

    # Check confidence collapse
    if confidence < policy.confidence_collapse_threshold:
        return TradeExit(
            reason="confidence_collapse",
            exit_price=current_price,
            realized_hold_min=elapsed_min,
            realized_rr=0.0,
            pnl_dollars=0.0,
        )

    # Check opposite signal
    if opposite_signal:
        return TradeExit(
            reason="opposite_signal",
            exit_price=current_price,
            realized_hold_min=elapsed_min,
            realized_rr=0.0,
            pnl_dollars=0.0,
        )

    return None


def check_expiry_exit(
    policy: ExecutionPolicy,
    trade_start_time: float,
    current_time: float,
    current_price: float,
    entry_price: float,
    stop_loss: float,
    direction: str,
    hold_target: HoldTarget,
) -> Optional[TradeExit]:
    """Check if trade expired (forced close at 15 min)."""
    validity = policy.validity_minutes * 60.0
    if current_time >= trade_start_time + validity:
        elapsed_min = (current_time - trade_start_time) / 60.0

        # Compute realized RR from entry to expiry
        if direction == "BUY":
            move_pct = (current_price - entry_price) / entry_price
        else:
            move_pct = (entry_price - current_price) / entry_price

        realized_rr = move_pct / abs((entry_price - stop_loss) / entry_price) if stop_loss != entry_price else 0.0

        return TradeExit(
            reason="expiry",
            exit_price=current_price,
            realized_hold_min=elapsed_min,
            realized_rr=realized_rr,
            pnl_dollars=0.0,
        )

    return None


def compute_realized_rr(
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    direction: str,
) -> float:
    """Compute realized risk:reward ratio."""
    if direction == "BUY":
        move = exit_price - entry_price
        risk = entry_price - stop_loss
    else:
        move = entry_price - exit_price
        risk = stop_loss - entry_price

    if risk <= 0:
        return 0.0

    return move / risk


def create_execution_policy() -> ExecutionPolicy:
    """Create default V27.1 execution policy."""
    return ExecutionPolicy()