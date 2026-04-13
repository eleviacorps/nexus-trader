"""
V24.1 Branch Realism Evaluation Script

This script evaluates branch realism for the V24.1 validation phase.
"""

import sys
import os
import numpy as np
from src.v24_1.branch_realism import BranchRealismEvaluator


def main():
    """Main function to run branch realism evaluation."""
    print("V24.1 Branch Realism Evaluation")
    print("=" * 35)

    # Initialize evaluator
    evaluator = BranchRealismEvaluator()

    # Create sample data for evaluation
    sample_branches = [
        np.random.randn(30, 36) for _ in range(64)  # 64 branches for evaluation
    ]

    # Evaluate branch realism
    print("Evaluating branch realism...")
    metrics = evaluator.evaluate_branch_realism(sample_branches)

    # Generate and display report
    report = evaluator.generate_realism_report(metrics)

    print("\nBranch Realism Evaluation Results:")
    print(f"  Branch Realism Score: {report['branch_realism_score']:.4f}")
    print(f"  Volatility Realism: {report['volatility_realism']:.4f}")
    print(f"  Analog Similarity: {report['analog_similarity']:.4f}")
    print(f"  Regime Consistency: {report['regime_consistency']:.4f}")
    print(f"  Path Plausibility: {report['path_plausibility']:.4f}")
    print(f"  Minority Usefulness: {report['minority_usefulness']:.4f}")
    print(f"  Cone Containment Rate: {report['cone_containment_rate']:.4f}")
    print(f"  Minority Rescue Rate: {report['minority_rescue_rate']:.4f}")

    # Save report to file
    output_dir = "outputs/v24_1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    report_file = os.path.join(output_dir, "branch_realism_report.json")
    with open(report_file, 'w') as f:
        import json
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\nBranch realism evaluation completed successfully!")
    else:
        print("\nBranch realism evaluation failed!")
        sys.exit(1)