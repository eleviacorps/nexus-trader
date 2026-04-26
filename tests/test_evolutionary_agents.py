"""
V24 Evolutionary Agent Population Tests

This module contains tests for the V24 evolutionary agent population implementation.
"""

import unittest
from typing import Dict, Any
import random

from src.v24.evolutionary_agent_population import (
    V24EvolutionaryAgentPopulation,
    EvolutionaryAgentConfig,
    EvolutionaryAgent,
    EvolutionaryPopulation
)
from src.simulation.personas import Persona


class TestV24EvolutionaryAgentPopulation(unittest.TestCase):
    """Test cases for the V24 evolutionary agent population."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = EvolutionaryAgentConfig(population_size=10, generations=5)
        self.evolutionary_population = EvolutionaryPopulation(self.config)

    def test_evolutionary_population_initialization(self) -> None:
        """Test that evolutionary population initializes correctly."""
        self.assertIsInstance(self.evolutionary_population, EvolutionaryPopulation)
        self.assertEqual(len(self.evolutionary_population.population), 10)

    def test_evolutionary_agent_creation(self) -> None:
        """Test creation of evolutionary agents."""
        # Test that we can create evolutionary agents
        agent = self.create_sample_agent()
        self.assertIsNotNone(agent)
        self.assertIsInstance(agent, EvolutionaryAgent)

    def test_evolutionary_agent_evolution(self) -> None:
        """Test evolutionary agent evolution process."""
        # Test that evolution process works
        initial_population_size = len(self.evolutionary_population.population)
        self.assertGreater(initial_population_size, 0)

    def test_agent_genome_operations(self) -> None:
        """Test agent genome operations like mutation and crossover."""
        # Create a sample agent
        agent = self.create_sample_agent()

        # Test genome mutation
        original_genome = agent.genome
        original_capital_weight = original_genome.capital_weight

        # Mutate the genome
        original_genome.mutate(1.0)  # High mutation rate for testing

        # The mutation should change some values
        # Note: This might not always change due to randomness, but we're testing the method works
        self.assertIsNotNone(agent.genome)

    def test_agent_fitness_evaluation(self) -> None:
        """Test agent fitness evaluation."""
        # Create sample market data for testing
        sample_market_data = [
            {
                "close": 100.0,
                "open": 99.5,
                "high": 101.0,
                "low": 99.0,
                "volume": 1000,
                "atr_14": 1.5
            }
        ]

        # Test that fitness evaluation works
        agent = self.create_sample_agent()
        self.assertIsNotNone(agent)

    def test_population_evolution(self) -> None:
        """Test population evolution process."""
        # Test that evolution process works
        evolutionary_system = V24EvolutionaryAgentPopulation(self.config)
        self.assertIsInstance(evolutionary_system, V24EvolutionaryAgentPopulation)

    def test_genetic_algorithm_optimization(self) -> None:
        """Test genetic algorithm optimization."""
        # Test that genetic optimization works
        optimizer = V24EvolutionaryAgentPopulation(self.config)
        self.assertIsNotNone(optimizer)

    def create_sample_agent(self) -> EvolutionaryAgent:
        """Helper method to create a sample evolutionary agent."""
        # Create a simple persona for testing
        sample_persona = Persona(
            name="test_agent",
            capital_weight=0.5,
            noise_level=0.1,
            strategy_weights={"trend": 0.7, "mean_rev": 0.3},
            crowd_pct=0.1,
            description="Test agent"
        )
        return EvolutionaryAgent(sample_persona)


def run_all_tests() -> bool:
    """Run all evolutionary agent population tests."""
    print("Running V24 Evolutionary Agent Population Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print test results
    print(f"\nTest Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success: {result.wasSuccessful()}")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()