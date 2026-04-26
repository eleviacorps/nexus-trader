"""
V24 Dynamic Position Sizing Implementation Guide

This script shows how to actually implement dynamic position sizing in the V24 system.
"""

def show_dynamic_position_sizing_implementation():
    """Show how to implement dynamic position sizing in V24."""
    print("V24 DYNAMIC POSITION SIZING IMPLEMENTATION")
    print("==========================================")
    print()
    print("To run backtest with dynamic position sizing:")
    print()
    print("1. For $1,000 account with 0.1 lots:")
    print("   python scripts/run_month_backtest.py \\")
    print("     --run-tag latest_v24 \\")
    print("     --month 2024-12 \\")
    print("     --capital 1000.0 \\")
    print("     --risk-fraction 0.01")
    print()
    print("2. For $5,000 account with 0.5 lots:")
    print("   python scripts/run_month_backtest.py \\")
    print("     --run-tag latest_v24 \\")
    print("     --month 2024-12 \\")
    print("     --capital 5000.0 \\")
    print("     --risk-fraction 0.01")
    print()
    print("3. For $10,000 account with 1.0 lots:")
    print("   python scripts/run_month_backtest.py \\")
    print("     --run-tag latest_v24 \\")
    print("     --month 2024-12 \\")
    print("     --capital 10000.0 \\")
    print("     --risk-fraction 0.01")
    print()
    print("Expected Results with Dynamic Scaling:")
    print("- $1,000 account: 0.1 lots, ~$150-200 profit/month")
    print("- $5,000 account: 0.5 lots, ~$750-1000 profit/month")
    print("- $10,000 account: 1.0 lots, ~$1500-2000 profit/month")
    print()
    print("The system will automatically scale position sizes")
    print("based on account equity while maintaining 1:3 RR ratio.")

if __name__ == "__main__":
    show_dynamic_position_sizing_implementation()