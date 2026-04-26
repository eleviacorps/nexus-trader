"""
V24 System Evaluation Guide

This script provides instructions for running actual backtests with your V24 system.
"""

def run_v24_backtest_evaluation():
    """Guide for running actual V24 backtest with your data."""

    print("V24 System Backtest Evaluation Guide")
    print("=" * 40)

    print("""
To run an actual evaluation of your V24 system:

1. Run the month bridge test:
   python scripts/run_v24_month_bridge.py --months "2023-12,2024-12" --meta-source auto

2. Or run the existing backtest framework:
   python scripts/run_month_backtest.py --run-tag your_model_tag --month 2024-12

3. For V24 specific evaluation, you can also use:
   python scripts/run_v24_month_bridge.py

The system will output comprehensive performance metrics including:
- Win rate
- Expected value correlation
- Sharpe ratio
- Trade frequency
- Cumulative returns
- Risk assessment scores

Sample command to run evaluation:
python scripts/run_v24_month_bridge.py --months "2023-12,2024-12" --meta-source auto
""")

    return True

if __name__ == "__main__":
    run_v24_backtest_evaluation()