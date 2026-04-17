"""
V24.4 Trade Cluster Filter
Prevents duplicate tactical trades within clusters.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class TradeClusterFilter:
    """Filter to prevent duplicate tactical trades within time/proximity clusters."""

    def __init__(self):
        self.trade_clusters = []
        self.cluster_threshold_minutes = 10
        self.cluster_entry_distance_threshold = 0.25  # in ATR units

    def filter_similar_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out similar trades within clusters.

        Args:
            trades: List of potential trades

        Returns:
            List of filtered trades
        """
        if not trades:
            return []

        # Sort trades by admission score (highest first)
        sorted_trades = sorted(trades, key=lambda x: x.get('admission_score', 0), reverse=True)

        # Filter out trades that are too similar
        filtered_trades = []
        for trade in sorted_trades:
            if not self._is_trade_in_cluster(trade, filtered_trades):
                filtered_trades.append(trade)

        return filtered_trades

    def _is_trade_in_cluster(self, new_trade: Dict[str, Any], existing_trades: List[Dict[str, Any]]) -> bool:
        """
        Check if a trade is in a cluster with existing trades.

        Args:
            new_trade: New trade to check
            existing_trades: List of existing trades

        Returns:
            bool: True if trade is in cluster
        """
        for existing_trade in existing_trades:
            # Check if trades are in same direction
            if new_trade.get('direction') != existing_trade.get('direction'):
                continue

            # Check time proximity
            new_time = new_trade.get('timestamp', datetime.now())
            existing_time = existing_trade.get('timestamp', datetime.now())
            time_diff = abs((new_time - existing_time).total_seconds() / 60)

            if time_diff < self.cluster_threshold_minutes:
                # Check entry proximity
                new_entry = new_trade.get('entry_price', 0)
                existing_entry = existing_trade.get('entry_price', 0)
                entry_diff = abs(new_entry - existing_entry)

                if entry_diff < self.cluster_entry_distance_threshold:
                    return True

        return False

    def should_filter_trade(self, trade: Dict[str, Any], trade_list: List[Dict[str, Any]]) -> bool:
        """
        Determine if a trade should be filtered out based on clustering.

        Args:
            trade: Trade to evaluate
            trade_list: List of existing trades

        Returns:
            bool: True if trade should be filtered out
        """
        for existing_trade in trade_list:
            # Check if trades are in same direction
            if trade.get('direction') != existing_trade.get('direction'):
                continue

            # Check time proximity
            trade_time = trade.get('timestamp', datetime.now())
            existing_time = existing_trade.get('timestamp', datetime.now())
            time_diff = abs((trade_time - existing_time).total_seconds() / 60)

            if time_diff < self.cluster_threshold_minutes:
                # Check entry proximity
                trade_entry = trade.get('entry_price', 0)
                existing_entry = existing_trade.get('entry_price', 0)
                entry_diff = abs(trade_entry - existing_entry)

                if entry_diff < self.cluster_entry_distance_threshold:
                    return True

        return False

    def add_to_cluster_tracking(self, trade: Dict[str, Any]):
        """
        Add trade to cluster tracking.

        Args:
            trade: Trade to add to tracking
        """
        self.trade_clusters.append({
            'timestamp': datetime.now(),
            'trade': trade
        })


def main():
    """Example usage of trade cluster filter."""
    # This would typically be integrated with the main trading system
    print("Trade Cluster Filter")
    print("This component prevents duplicate tactical trades within clusters.")


if __name__ == "__main__":
    main()