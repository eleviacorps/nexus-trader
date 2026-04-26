"""V27.1 Risk Manager with progressive risk stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskStage:
    """Risk stage configuration."""
    name: str
    risk_percent: float
    max_consecutive_losses: int


RISK_STAGES = [
    RiskStage(name="stage_1", risk_percent=0.25, max_consecutive_losses=3),
    RiskStage(name="stage_2", risk_percent=0.50, max_consecutive_losses=2),
    RiskStage(name="stage_3", risk_percent=0.75, max_consecutive_losses=2),
    RiskStage(name="stage_4", risk_percent=1.00, max_consecutive_losses=1),
]


class RiskManager:
    """Manages position sizing based on account risk."""

    def __init__(
        self,
        account_balance: float,
        stage: int = 1,
    ):
        self.account_balance = account_balance
        self.stage = max(1, min(stage, 4))
        self.consecutive_losses = 0

        stage_config = RISK_STAGES[self.stage - 1]
        self.risk_percent = stage_config.risk_percent
        self.max_consecutive_losses = stage_config.max_consecutive_losses

    def compute_lot_size(
        self,
        entry_price: float,
        stop_loss: float,
        pip_value: float = 1.0,
    ) -> float:
        """Compute lot size based on risk percent.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            pip_value: Value per pip/unit

        Returns:
            Lot size to trade
        """
        risk_dollars = self.account_balance * (self.risk_percent / 100.0)

        # Stop distance in price terms
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0:
            return 0.0

        # Lot size = risk_dollars / stop_distance
        lot_size = risk_dollars / stop_distance

        return lot_size

    def compute_lot_size_from_pips(
        self,
        entry_price: float,
        stop_loss_pips: float,
        pip_size: float = 0.0001,
        pip_value: float = 1.0,
    ) -> float:
        """Compute lot size from stop loss in pips.

        Args:
            entry_price: Entry price
            stop_loss_pips: Stop loss in pips
            pip_size: Size of one pip
            pip_value: Dollar value per pip per lot

        Returns:
            Lot size to trade
        """
        risk_dollars = self.account_balance * (self.risk_percent / 100.0)

        if stop_loss_pips <= 0:
            return 0.0

        risk_per_lot = stop_loss_pips * pip_value
        lot_size = risk_dollars / risk_per_lot

        return lot_size

    def on_win(self):
        """Called when trade wins."""
        self.consecutive_losses = 0

    def on_loss(self):
        """Called when trade loses."""
        self.consecutive_losses += 1

        # Check if should downgrade stage
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.downgrade_stage()

    def upgrade_stage(self):
        """Move to higher risk stage."""
        if self.stage < 4:
            self.stage += 1
            self._apply_stage()

    def downgrade_stage(self):
        """Move to lower risk stage."""
        if self.stage > 1:
            self.stage -= 1
            self._apply_stage()
        self.consecutive_losses = 0

    def _apply_stage(self):
        """Apply current stage settings."""
        stage_config = RISK_STAGES[self.stage - 1]
        self.risk_percent = stage_config.risk_percent
        self.max_consecutive_losses = stage_config.max_consecutive_losses

    def get_risk_dollars(self) -> float:
        """Get dollar amount at risk for current stage."""
        return self.account_balance * (self.risk_percent / 100.0)


def create_risk_manager(account_balance: float, stage: int = 1) -> RiskManager:
    """Create risk manager with specified stage."""
    return RiskManager(account_balance=account_balance, stage=stage)