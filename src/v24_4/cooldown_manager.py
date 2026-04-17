"""
V24.4 Tactical Cooldown Manager
Implements adaptive threshold adjustments based on recent performance.
"""
from typing import List, Dict, Any
from collections import deque


class CooldownManager:
    """Manages tactical trading cooldowns and threshold adjustments."""

    def __init__(self):
        self.recent_results = deque(maxlen=5)  # Track last 5 trades
        self.current_threshold_adjustment = 0.0
        self.base_threshold = 0.7
        self.max_threshold = 0.8
        self.min_threshold = 0.6

    def record_trade_result(self, was_profitable: bool):
        """
        Record the result of a recent trade.

        Args:
            was_profitable: Whether the trade was profitable
        """
        self.recent_results.append(was_profitable)
        self._adjust_threshold_based_on_streak()

    def _adjust_threshold_based_on_streak(self):
        """Adjust threshold based on recent performance streak."""
        if len(self.recent_results) >= 3:
            recent = list(self.recent_results)
            if len(recent) >= 3:
                # Check for 3 consecutive wins
                if all(recent[-3:]):
                    self.current_threshold_adjustment = -0.02  # Relax threshold after wins
                # Check for 2 consecutive losses
                elif len(recent) >= 2 and not any(recent[-2:]):
                    self.current_threshold_adjustment = 0.05  # Tighten threshold after losses

    def get_current_threshold(self, base_threshold: float = 0.7) -> float:
        """
        Get current adjusted threshold.

        Args:
            base_threshold: Base threshold value

        Returns:
            float: Adjusted threshold
        """
        adjusted = base_threshold + self.current_threshold_adjustment
        return max(self.min_threshold, min(self.max_threshold, adjusted))

    def should_enter_cooldown(self) -> bool:
        """
        Determine if system should enter cooldown based on recent losses.

        Returns:
            bool: Whether to enter cooldown
        """
        if len(self.recent_results) >= 2:
            recent = list(self.recent_results)
            return not any(recent[-2:])  # Two consecutive losses
        return False


def main():
    """Example usage of cooldown manager."""
    # This would typically be integrated with the main trading system
    print("Cooldown Manager")
    print("This component manages tactical trading cooldowns and threshold adjustments.")


if __name__ == "__main__":
    main()