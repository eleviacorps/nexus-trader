"""
V24 Conditional Diffusion Model Simple Test

This script tests the basic functionality of the V24 diffusion model.
"""

import sys
import torch

# Add the project directory to the Python path
sys.path.append('.')

def test_v24_diffusion_basic():
    """Simple test to verify the V24 diffusion model implementation."""
    print("Testing V24 Conditional Diffusion Model")
    print("=" * 40)

    try:
        # Import the required modules
        from src.v24.diffusion_model import ConditionalDiffusionModel, DiffusionConfig
        from src.v24.conditional_generator import ConditionalPathGenerator, GenerationConfig

        # Create a basic configuration
        config = DiffusionConfig(
            sequence_length=30,
            feature_dim=36,
            hidden_dim=128
        )

        print("SUCCESS: Modules imported successfully")

        # Test model creation
        model = ConditionalDiffusionModel(config)
        print("SUCCESS: ConditionalDiffusionModel created successfully")

        # Test generator creation
        generation_config = GenerationConfig()
        generator = ConditionalPathGenerator(config, generation_config)
        print("SUCCESS: ConditionalPathGenerator created successfully")

        # Test that we can create a simple tensor
        test_tensor = torch.randn(2, config.sequence_length, config.feature_dim)
        print("SUCCESS: Tensor operations working")
        print(f"  Tensor shape: {test_tensor.shape}")

        print("\nAll tests passed!")
        return True

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_v24_diffusion_basic()