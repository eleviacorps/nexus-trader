"""
V24.3 Stability Testing
Script to test system stability across multiple runs with different parameters.
"""
import pandas as pd
import numpy as np
import json
import random
from typing import Dict, Any, List
from src.v24_3.tactical_router import TacticalRouter
from src.v24_3.execution_simulator import ExecutionSimulator


class StabilityTester:
    """Test system stability across multiple runs with different parameters."""

    def __init__(self):
        self.tactical_router = TacticalRouter()
        self.execution_simulator = ExecutionSimulator()
        self.test_results = []

    def run_stability_test(self, num_runs: int = 10) -> Dict[str, Any]:
        """
        Run stability test with multiple iterations using different parameters.

        Args:
            num_runs (int): Number of test runs to perform

        Returns:
            dict: Stability test results
        """
        print(f"Running stability test with {num_runs} iterations...")

        results = []
        for run in range(num_runs):
            print(f"Running test iteration {run + 1}/{num_runs}")

            # Generate different parameters for each run
            test_params = self._generate_test_parameters(run)

            # Run system with these parameters
            run_result = self._run_single_test(test_params)
            results.append(run_result)

            print(f"  Win Rate: {run_result.get('win_rate', 0):.2%}")
            print(f"  Expectancy: {run_result.get('expectancy', 0):.4f}R")
            print(f"  Participation: {run_result.get('participation_rate', 0):.2%}")

        # Analyze overall stability
        stability_analysis = self._analyze_stability(results)

        # Save results
        self._save_stability_report(results, stability_analysis)

        return stability_analysis

    def _generate_test_parameters(self, run_id: int) -> Dict[str, Any]:
        """Generate different parameters for each test run."""
        # Different random seeds for each run
        random_seed = 42 + run_id

        # Different market conditions
        market_volatility = np.random.uniform(0.5, 2.0)

        # Different data slices
        data_start_offset = run_id * 100  # Different starting point in data

        return {
            'random_seed': random_seed,
            'market_volatility': market_volatility,
            'data_start_offset': data_start_offset,
            'branch_initialization_seed': random_seed * 2
        }

    def _run_single_test(self, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test with given parameters."""
        # Set random seed for reproducibility
        np.random.seed(test_params['random_seed'])
        random.seed(test_params['random_seed'])

        # Generate mock market data with specified volatility
        market_data = self._generate_mock_market_data(
            volatility=test_params['market_volatility']
        )

        # Run multiple trades to gather statistics
        trades = []
        total_trades = 0
        winning_trades = 0
        total_expectancy = 0.0
        total_participation = 0

        # Simulate 100 trading decisions
        for i in range(100):
            # Generate strategic signal
            strategic_signal = self._generate_mock_strategic_signal()

            # Route trade through tactical router
            routing_decision = self.tactical_router.route_trade(market_data, strategic_signal)

            # Check if trade was taken
            if routing_decision['final_decision']['should_trade']:
                total_trades += 1
                total_participation += 1

                # Simulate trade outcome
                trade_outcome = self._simulate_trade_outcome(
                    routing_decision['final_decision']['confidence']
                )

                if trade_outcome > 0:
                    winning_trades += 1

                total_expectancy += trade_outcome

        # Calculate metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_expectancy = total_expectancy / 100 if total_trades > 0 else 0  # Simplified
        participation_rate = total_participation / 100

        return {
            'run_id': len(self.test_results),
            'win_rate': win_rate,
            'expectancy': avg_expectancy,
            'participation_rate': participation_rate,
            'total_trades': total_trades,
            'parameters': test_params
        }

    def _generate_mock_market_data(self, volatility: float = 1.0) -> pd.DataFrame:
        """Generate mock market data with specified volatility."""
        # Generate price series with specified volatility
        base_price = 1850.0
        prices = [base_price]

        # Generate 1000 price points
        for i in range(999):
            # Random walk with volatility scaling
            change = np.random.normal(0, 0.5 * volatility)
            prices.append(prices[-1] + change)

        # Create dates for the same length
        dates = pd.date_range('2026-01-01', periods=len(prices), freq='1min')

        # Make sure all arrays have the same length
        data_length = len(prices)

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + np.random.random() * 2 for p in prices],
            'low': [p - np.random.random() * 2 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, data_length)
        })

    def _generate_mock_strategic_signal(self) -> Dict[str, Any]:
        """Generate mock strategic signal."""
        signals = ['buy', 'sell', 'hold']
        signal = np.random.choice(signals, p=[0.3, 0.3, 0.4])

        return {
            'signal': signal,
            'confidence': np.random.random(),
            'reason': 'Generated by strategic engine'
        }

    def _simulate_trade_outcome(self, confidence: float) -> float:
        """Simulate trade outcome based on confidence."""
        # Higher confidence trades have better outcomes
        expected_return = confidence * 0.02  # 2% max expected return
        actual_return = np.random.normal(expected_return, 0.01)  # Add some variance
        return actual_return

    def _analyze_stability(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stability across multiple test runs."""
        if not results:
            return {'message': 'No test results to analyze'}

        # Extract metrics for analysis
        win_rates = [r.get('win_rate', 0) for r in results]
        expectancies = [r.get('expectancy', 0) for r in results]
        participation_rates = [r.get('participation_rate', 0) for r in results]

        # Calculate statistics
        mean_win_rate = np.mean(win_rates)
        std_win_rate = np.std(win_rates)
        mean_expectancy = np.mean(expectancies)
        std_expectancy = np.std(expectancies)
        mean_participation = np.mean(participation_rates)
        std_participation = np.std(participation_rates)

        # Check stability criteria
        stable_win_rate = 0.60 <= mean_win_rate <= 0.68
        stable_expectancy = 0.20 <= mean_expectancy <= 0.30
        stable_participation = 0.05 <= mean_participation <= 0.15

        stability_result = {
            'overall_stability': stable_win_rate and stable_expectancy and stable_participation,
            'metrics': {
                'mean_win_rate': mean_win_rate,
                'std_win_rate': std_win_rate,
                'mean_expectancy': mean_expectancy,
                'std_expectancy': std_expectancy,
                'mean_participation_rate': mean_participation,
                'std_participation_rate': std_participation
            },
            'stability_checks': {
                'win_rate_stable': stable_win_rate,
                'expectancy_stable': stable_expectancy,
                'participation_stable': stable_participation
            },
            'individual_results': results
        }

        return stability_result

    def _save_stability_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save stability test results to file."""
        # Convert analysis results to JSON-serializable format
        report = {
            'test_results': [
                {
                    'run_id': r.get('run_id', 0),
                    'win_rate': float(r.get('win_rate', 0)),
                    'expectancy': float(r.get('expectancy', 0)),
                    'participation_rate': float(r.get('participation_rate', 0)),
                    'total_trades': r.get('total_trades', 0),
                    'parameters': r.get('parameters', {})
                }
                for r in results
            ],
            'analysis': {
                'overall_stability': bool(analysis.get('overall_stability', False)),
                'metrics': analysis.get('metrics', {}),
                'stability_checks': {
                    'win_rate_stable': bool(analysis.get('stability_checks', {}).get('win_rate_stable', False)),
                    'expectancy_stable': bool(analysis.get('stability_checks', {}).get('expectancy_stable', False)),
                    'participation_stable': bool(analysis.get('stability_checks', {}).get('participation_stable', False))
                }
            }
        }

        # Save to file
        with open('outputs/v24_3/stability_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("Stability test report generated and saved.")


def main():
    """Example usage of the stability tester."""
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs/v24_3', exist_ok=True)

    # Initialize stability tester
    tester = StabilityTester()

    print("Stability Tester initialized.")
    print("Ready to run stability tests.")

    # In a real scenario, you would run the full test
    # For this example, we'll just show the setup
    print("To run full stability test, call:")
    print("  tester.run_stability_test(num_runs=10)")


if __name__ == "__main__":
    main()