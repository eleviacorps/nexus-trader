"""
V24 Running with Larger Position Sizes

This script shows how to run with larger position sizes.
"""

def run_with_larger_positions():
    """Show how to run with larger position sizes."""
    print("RUNNING WITH LARGER POSITION SIZES")
    print("=================================")
    print()
    print("To run with larger position sizes:")
    print()
    print("1. For 0.2 lots:")
    print("   python scripts/run_v24_month_bridge.py \\")
    print("     --months \"2024-12\" \\")
    print("     --meta-source auto")
    print()
    print("2. For 0.5 lots:")
    print("   (This would require custom parameter adjustment)")
    print()
    print("3. For 1.0 lots:")
    print("   (This would require custom parameter adjustment)")
    print()
    print("The system likely uses conservative position sizing")
    print("by default to protect capital. Larger positions would")
    print("give higher returns but higher risk.")

if __name__ == "__main__":
    run_with_larger_positions()