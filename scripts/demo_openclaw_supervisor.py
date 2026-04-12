"""
V24 OpenClaw Supervisor Demo

This script demonstrates the V24 OpenClaw supervisor implementation.
"""

import sys
import os

# Add the project directory to the Python path
sys.path.append('.')

from src.v24.openclaw_supervisor import (
    V24OpenClawSupervisor,
    SupervisorConfig
)


def demo_openclaw_supervisor():
    """Demo script to show the V24 OpenClaw supervisor in action."""
    print("V24 OpenClaw Supervisor Demo")
    print("=" * 50)

    try:
        # Create configuration
        config = SupervisorConfig()
        print("Creating OpenClaw supervisor...")

        # Create supervisor system
        supervisor = V24OpenClawSupervisor(config)
        print("SUCCESS: OpenClaw supervisor created")

        # Test basic functionality
        print("\nTesting supervisor functionality...")

        # Start the supervisor
        supervisor.start()
        print("SUCCESS: Supervisor started")

        # Test system metrics collection
        metrics = supervisor.supervisor.collect_system_metrics()
        print(f"SUCCESS: System metrics collected")
        print(f"  CPU Usage: {metrics.cpu_usage}%")
        print(f"  Memory Usage: {metrics.memory_usage}%")

        # Test system health evaluation
        health = supervisor.supervisor.evaluate_system_health()
        print(f"SUCCESS: System health evaluated: {health}")

        # Test system report generation
        report = supervisor.generate_system_report()
        print(f"SUCCESS: System report generated")
        print(f"  System Status: {report.get('system_status', 'UNKNOWN')}")

        # Test agent monitoring
        agents = supervisor.supervisor.monitor_agents()
        print(f"SUCCESS: Agent monitoring completed")
        print(f"  Active Agents: {len(agents)}")

        # Stop the supervisor
        supervisor.stop()
        print("SUCCESS: Supervisor stopped")

        print("\nDemo completed successfully!")
        print("Phase 7 OpenClaw Supervisor implementation is ready for integration.")
        return True

    except Exception as e:
        print(f"Demo failed with error: {e}")
        return False


if __name__ == "__main__":
    demo_openclaw_supervisor()