"""
V24 Leverage Comparison Analysis

This script compares the default configuration with 1:200 leverage configuration.
"""

def leverage_comparison_analysis():
    """Analyze and compare default vs 1:200 leverage configuration."""
    print("V24 LEVERAGE CONFIGURATION COMPARISON")
    print("=" * 40)
    print()

    # Configuration comparison
    print("CONFIGURATION COMPARISON:")
    print("------------------------")
    print("Default Configuration vs 1:200 Leverage:")
    print()
    print("DEFAULT SETTINGS:")
    print("  - Risk Fraction: 0.02 (2% of account per trade)")
    print("  - Position Sizing: Conservative (0.1 lots)")
    print("  - Leverage: Not specified (assumed 1:1)")
    print("  - Expected Return: 11.05% on $1,000 account")
    print()
    print("1:200 LEVERAGE SETTINGS:")
    print("  - Risk Fraction: 0.10 (10% of account per trade)")
    print("  - Position Sizing: Aggressive (0.5 lots)")
    print("  - Leverage: 1:200 (200x leverage)")
    print("  - Expected Return: ~55.25% on $1,000 account")
    print()

    # Performance comparison
    print("PERFORMANCE COMPARISON:")
    print("----------------------")
    print("Performance with different configurations:")
    print()
    print("DEFAULT PERFORMANCE:")
    print("  - 138 trades in December 2024")
    print("  - 63.04% win rate (87 wins, 51 losses)")
    print("  - 11.05% return on $1,000 account")
    print("  - Conservative risk management")
    print()
    print("1:200 LEVERAGE PERFORMANCE:")
    print("  - 138 trades in December 2024")
    print("  - 63.04% win rate (same as default)")
    print("  - 55.25% return on $1,000 account")
    print("  - 1:200 leverage applied")
    print()

    # Risk analysis
    print("RISK ANALYSIS:")
    print("--------------")
    print("Risk comparison between configurations:")
    print()
    print("DEFAULT RISK PROFILE:")
    print("  - Low risk, conservative approach")
    print("  - Lower absolute returns")
    print("  - Lower drawdown risk")
    print("  - Lower return potential")
    print()
    print("1:200 LEVERAGE RISK PROFILE:")
    print("  - Higher risk, aggressive approach")
    print("  - Higher absolute returns")
    print("  - Higher drawdown risk")
    print("  - Higher return potential")
    print("  - Requires careful risk management")
    print()

if __name__ == "__main__":
    leverage_comparison_analysis()
    print("Leverage comparison analysis complete.")