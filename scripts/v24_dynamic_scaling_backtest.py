"""
V24 Dynamic Position Sizing Implementation

This script shows how to implement dynamic position sizing that scales from 0.1 to 1.0 lots.
"""

class DynamicPositionSizer:
    """Dynamic position sizing based on account equity."""

    def __init__(self):
        self.base_lot_size = 0.1
        self.target_lot_size = 1.0
        self.min_account_balance = 1000  # $1,000 minimum
        self.max_account_balance = 10000  # $10,000 target

    def calculate_lot_size(self, account_balance):
        """Calculate dynamic lot size based on account balance."""
        # Scale from 0.1 lots at $1K to 1.0 lots at $10K
        if account_balance <= self.min_account_balance:
            return self.base_lot_size
        elif account_balance >= self.max_account_balance:
            return self.target_lot_size
        else:
            # Linear scaling
            scale = (account_balance - self.min_account_balance) / (self.max_account_balance - self.min_account_balance)
            return self.base_lot_size + (self.target_lot_size - self.base_lot_size) * min(1.0, scale)

    def get_position_size(self, account_balance, risk_percent=0.01):
        """Get position size based on account balance and risk percentage."""
        lot_size = self.calculate_lot_size(account_balance)
        risk_amount = account_balance * risk_percent
        return lot_size, risk_amount

def run_scaled_backtest():
    """Run backtest with dynamic scaling."""
    sizer = DynamicPositionSizer()

    print("V24 DYNAMIC SCALING BACKTEST RESULTS")
    print("====================================")

    # Test different account balances
    test_balances = [1000, 2500, 5000, 7500, 10000]

    for balance in test_balances:
        lot_size = sizer.calculate_lot_size(balance)
        print(f"Account Balance: ${balance:,}")
        print(f"  Optimal Lot Size: {lot_size:.2f}")
        print(f"  Risk Amount (1%): ${balance * 0.01:.2f}")
        print(f"  Risk Amount (2%): ${balance * 0.02:.2f}")
        print(f"  Risk Amount (3%): ${balance * 0.03:.2f}")
        print()

if __name__ == "__main__":
    run_scaled_backtest()