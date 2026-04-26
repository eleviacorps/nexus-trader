"""
V24 OpenClaw Supervisor Tests

This module contains tests for the V24 OpenClaw supervisor implementation.
"""

import unittest
from datetime import datetime
from src.v24.openclaw_supervisor import (
    V24OpenClawSupervisor,
    SupervisorConfig,
    SystemMetrics,
    AgentStatus
)


class TestV24OpenClawSupervisor(unittest.TestCase):
    """Test cases for the V24 OpenClaw supervisor."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = SupervisorConfig()
        self.supervisor = V24OpenClawSupervisor(self.config)

    def test_supervisor_initialization(self) -> None:
        """Test that supervisor initializes correctly."""
        self.assertIsNotNone(self.supervisor)
        self.assertIsInstance(self.supervisor, V24OpenClawSupervisor)

    def test_system_metrics_collection(self) -> None:
        """Test system metrics collection."""
        metrics = self.supervisor.supervisor.collect_system_metrics()
        self.assertIsInstance(metrics, SystemMetrics)

    def test_agent_monitoring(self) -> None:
        """Test agent monitoring functionality."""
        agents = self.supervisor.supervisor.monitor_agents()
        self.assertIsInstance(agents, dict)

    def test_system_health_evaluation(self) -> None:
        """Test system health evaluation."""
        health = self.supervisor.supervisor.evaluate_system_health()
        self.assertIn(health, ["CRITICAL", "WARNING", "HEALTHY"])

    def test_alert_generation(self) -> None:
        """Test alert generation."""
        alerts = self.supervisor.supervisor.generate_alerts()
        self.assertIsInstance(alerts, list)

    def test_system_coordination(self) -> None:
        """Test system coordination."""
        # Test that coordination functions work
        self.supervisor.supervisor.coordinate_phases()
        self.assertTrue(True)

    def test_system_report_generation(self) -> None:
        """Test system report generation."""
        report = self.supervisor.generate_system_report()
        self.assertIsInstance(report, dict)


def run_all_tests() -> bool:
    """Run all OpenClaw supervisor tests."""
    print("Running V24 OpenClaw Supervisor Tests")
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