"""
V24 Custom Backtest Configuration

This script shows how to configure a backtest with your specific parameters.
"""

def show_custom_backtest_setup():
    """Show how to set up a custom backtest with specific parameters."""
    print("V24 Custom Backtest Configuration")
    print("=" * 40)
    print("Parameters:")
    print("- Account Balance: $1,000")
    print("- Leverage: 1:200")
    print("- Risk/Reward Ratio: TP 3x SL")
    print("- Position Sizing: Maximum frequency trading")
    print()
    print("To run this backtest, you would use:")
    print()
    print("1. V24 Month Bridge Test:")
    print("   python scripts/run_v24_month_bridge.py \\")
    print("     --months \"2023-12,2024-12\" \\")
    print("     --meta-source auto")
    print()
    print("2. Custom Backtest with Capital Parameters:")
    print("   python scripts/run_month_backtest.py \\")
    print("     --run-tag latest_v24 \\")
    print("     --month 2024-12 \\")
    print("     --capital 1000.0 \\")
    print("     --risk-fraction 0.005 \\")
    print("     --horizon 15m")
    print()
    print("Expected Configuration:")
    print("- Position size: Based on 1:200 leverage")
    print("- Stop Loss: 1 unit")
    print("- Take Profit: 3 units (1:3 ratio)")
    print("- Risk per trade: 0.5% of capital ($5)")
    print("- Maximum trade frequency: As many as signals allow")

if __name__ == "__main__":
    show_custom_backtest_setup()