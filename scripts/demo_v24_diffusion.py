"""
V24 Conditional Diffusion Model Demo

This script demonstrates the V24 conditional diffusion model implementation.
"""

import sys
import torch

# Add the project directory to the Python path
sys.path.append('.')

from src.v24.diffusion_model import ConditionalDiffusionModel, DiffusionConfig
from src.v24.conditional_generator import ConditionalPathGenerator, GenerationConfig

def demo_diffusion_model():
    """Demo script to show the V24 diffusion model implementation."""
    print("V24 Conditional Diffusion Model Demo")
    print("=" * 40)

    # Create a basic configuration
    config = DiffusionConfig(
        sequence_length=30,
        feature_dim=36,
        hidden_dim=128
    )

    try:
        # Test model creation
        print("Creating ConditionalDiffusionModel...")
        model = ConditionalDiffusionModel(config)
        print("✓ ConditionalDiffusionModel created successfully")
        print(f"  Model type: {type(model)}")

        # Test generator creation
        print("\nCreating ConditionalPathGenerator...")
        generation_config = GenerationConfig()
        generator = ConditionalPathGenerator(config, generation_config)
        print("✓ ConditionalPathGenerator created successfully")
        print(f"  Generator type: {type(generator)}")

        # Test that we can create a simple tensor
        print("\nTesting tensor operations...")
        test_tensor = torch.randn(2, config.sequence_length, config.feature_dim)
        print("✓ Tensor operations working")
        print(f"  Tensor shape: {test_tensor.shape}")

        print("\nDemo completed successfully!")
        return True

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        return False

if __name__ == "__main__":
    demo_diffusion_model()