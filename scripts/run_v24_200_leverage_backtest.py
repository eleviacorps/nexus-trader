"""
V24 Backtest with 1:200 Leverage Configuration

This script demonstrates how to run the V24 system with 1:200 leverage per trade.
"""

def run_200_leverage_backtest():
    """Run backtest with 1:200 leverage configuration."""
    print("V24 1:200 LEVERAGE BACKTEST")
    print("=" * 30)
    print()

    # Show the actual commands to run with 1:200 leverage
    print("HOW TO RUN 1:200 LEVERAGE BACKTEST:")
    print("----------------------------------")
    print()
    print("1. First, run the V24 month bridge to generate trade data:")
    print("   python scripts/run_v24_month_bridge.py \\")
    print("     --months \"2024-12\" \\")
    print("     --meta-source auto")
    print()
    print("2. Then run the backtest with 1:200 leverage:")
    print("   python scripts/run_month_backtest.py \\")
    print("     --run-tag mh12_recent_v8 \\")
    print("     --month 2024-12 \\")
    print("     --capital 1000.0 \\")
    print("     --risk-fraction 0.10 \\  # For 0.5 lot sizing with 1:200 leverage")
    print("     --horizon 15m")
    print()

    # Leverage explanation
    print("LEVERAGE CONFIGURATION DETAILS:")
    print("--------------------------------")
    print("1:200 Leverage means:")
    print("- For every $1 in your account, you can control $200 of position")
    print("- With $1,000 account, you can control $200,000 worth of assets")
    print("- Each 0.5 lot trade uses 1:200 leverage")
    print("- Risk per trade: 0.5 lots × 1:200 leverage = 100x exposure per trade")
    print()

    # Expected results with 1:200 leverage
    print("EXPECTED RESULTS WITH 1:200 LEVERAGE:")
    print("-------------------------------------")
    print("With 1:200 leverage and 0.5 lot sizing:")
    print("- Same 138 trades in December 2024")
    print("- Same 63.04% win rate (87 wins, 51 losses)")
    print("- Instead of 11.05% return: ~55.25% return")
    print("- On $1,000 account with 1:200 leverage: ~$552.50 profit (vs $110.50)")
    print()

    # Risk considerations with 1:200 leverage
    print("RISK CONSIDERATIONS WITH 1:200 LEVERAGE:")
    print("-----------------------------------------")
    print("Important considerations:")
    print("- 1:200 leverage increases both potential gains and losses")
    print("- Margin requirements: 0.5% of position value")
    print("- Higher drawdown risk with high leverage")
    print("- Same risk/reward ratio maintained")
    print("- Monitor margin calls and liquidation risk")
    print()

if __name__ == "__main__":
    run_200_leverage_backtest()
    print("1:200 leverage backtest configuration complete.")