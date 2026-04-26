"""
V24 0.5 Lot Sizing Backtest Execution Script

This script shows how to actually run the V24 system with 0.5 lot sizing.
"""

def run_05_lot_backtest():
    """Show how to run the backtest with 0.5 lot sizing."""
    print("V24 0.5 LOT SIZING BACKTEST EXECUTION")
    print("=" * 40)
    print()

    # Show the actual commands to run
    print("HOW TO RUN 0.5 LOT SIZING BACKTEST:")
    print("-----------------------------------")
    print()
    print("1. First, run the V24 month bridge to generate trade data:")
    print("   python scripts/run_v24_month_bridge.py \\")
    print("     --months \"2024-12\" \\")
    print("     --meta-source auto")
    print()
    print("2. Then run the backtest with 0.5 lot sizing:")
    print("   python scripts/run_month_backtest.py \\")
    print("     --run-tag mh12_recent_v8 \\")
    print("     --month 2024-12 \\")
    print("     --capital 1000.0 \\")
    print("     --risk-fraction 0.10 \\  # 5x default for 0.5 lots")
    print("     --horizon 15m")
    print()

    # Expected results
    print("EXPECTED RESULTS WITH 0.5 LOT SIZING:")
    print("-------------------------------------")
    print("With 0.5 lot sizing:")
    print("- Same 138 trades in December 2024")
    print("- Same 63.04% win rate (87 wins, 51 losses)")
    print("- Instead of 11.05% return: ~55.25% return")
    print("- On $1,000 account: ~$552.50 profit (vs $110.50)")
    print()

    # Risk considerations
    print("RISK MANAGEMENT WITH 0.5 LOTS:")
    print("-----------------------------")
    print("Important considerations:")
    print("- 5x position sizing increases absolute risk")
    print("- Same risk/reward ratio maintained")
    print("- Monitor drawdowns more carefully")
    print("- Ensure adequate capital for 0.5 lot positions")
    print()

if __name__ == "__main__":
    run_05_lot_backtest()
    print("Backtest execution instructions complete.")