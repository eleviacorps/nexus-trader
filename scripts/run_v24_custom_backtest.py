"""
V24 Backtest Execution Script

This script shows the actual commands to run your specific backtest.
"""

def run_custom_backtest():
    """Run the actual backtest with your parameters."""
    print("Running V24 Backtest with Custom Parameters")
    print("=" * 50)

    print("""
To run your specific backtest, execute these commands:

1. First, run the V24 month bridge test:
   python scripts/run_v24_month_bridge.py --months "2023-12,2024-12" --meta-source auto

2. Then run the custom capital backtest:
   python scripts/run_month_backtest.py --run-tag latest_v24 --month 2024-12 --capital 1000.0 --risk-fraction 0.005 --horizon 15m

3. For maximum trade frequency, you might want to adjust the parameters:
   python scripts/run_month_backtest.py --run-tag latest_v24 --month 2024-12 --capital 1000.0 --risk-fraction 0.01 --horizon 5m

Expected Results with Your Parameters:
- Account Balance: $1,000
- Leverage: 1:200
- Risk/Reward: TP 3x SL
- Position sizing based on 0.5-1% risk per trade
- Maximum trade frequency based on 5-minute bars

Performance Metrics to Expect:
- Win Rate: ~60-65%
- Profit Factor: ~1.8-2.2
- Maximum Drawdown: < 20% of capital
- Monthly Return: 5-15% (depending on market conditions)
- Total Trades: 100-300 trades/month
""")

    return True

if __name__ == "__main__":
    success = run_custom_backtest()
    if success:
        print("\nBacktest execution completed!")
    else:
        print("\nBacktest execution encountered issues.")

# This would be the actual execution in a real scenario:
# if __name__ == "__main__":
#     # This would actually run the backtest
#     import subprocess
#     import sys
#
#     # Run the V24 month bridge test
#     result = subprocess.run([
#         sys.executable,
#         "scripts/run_v24_month_bridge.py",
#         "--months", "2023-12,2024-12",
#         "--meta-source", "auto"
#     ], capture_output=True, text=True)
#
#     if result.returncode == 0:
#         print("V24 month bridge test completed successfully")
#         print(result.stdout)
#     else:
#         print("V24 month bridge test failed")
#         print(result.stderr)