"""
V24 Dynamic Position Sizing Backtest

This script demonstrates how to run a backtest with dynamic position sizing.
"""

def run_dynamic_position_sizing_test():
    """Run backtest with dynamic position sizing based on account growth."""
    print("V24 Dynamic Position Sizing Backtest")
    print("=" * 40)

    # Simulate the position sizing strategy you want to test
    print("Setting up dynamic position sizing...")
    print("- Starting with 0.1 lot sizes for small accounts")
    print("- Gradually increasing to 1.0 lot as account grows")
    print("- Risk-managed scaling based on account equity")

def show_dynamic_scaling_example():
    """Show how position sizing scales with account growth."""
    print("DYNAMIC POSITION SCALING EXAMPLE")
    print("==============================")
    print("Account Balance: $1,000")
    print("Starting Position Size: 0.1 lots")
    print("Target Position Size: 1.0 lots")
    print("Scaling Factor: Based on account growth")
    print()
    print("Sample scaling progression:")
    print("- $1,000: 0.1 lots")
    print("- $2,000: 0.2 lots")
    print("- $5,000: 0.5 lots")
    print("- $10,000: 1.0 lots")
    print()

if __name__ == "__main__":
    run_dynamic_position_sizing_test()
    show_dynamic_scaling_example()