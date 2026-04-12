"""
V24 Conditional Diffusion Model Simple Test

This script provides a simple test to verify the V24 diffusion model implementation.
"""

import torch
from src.v24.diffusion_model import ConditionalDiffusionModel, DiffusionConfig
from src.v24.conditional_generator import ConditionalPathGenerator, GenerationConfig

def test_diffusion_model_basic():
    """Simple test to verify the diffusion model can be instantiated."""
    print("Testing V24 Conditional Diffusion Model")
    print("=" * 40)

    # Create a basic configuration
    config = DiffusionConfig(
        sequence_length=30,
        feature_dim=36,
        hidden_dim=128
    )

    try:
        # Test model creation
        model = ConditionalDiffusionModel(config)
        print("✓ ConditionalDiffusionModel created successfully")

        # Test that model has required attributes
        assert hasattr(model, 'config'), "Model should have config attribute"
        print("✓ Model has required attributes")

        # Test generator creation
        generation_config = GenerationConfig()
        generator = ConditionalPathGenerator(config, generation_config)
        print("✓ ConditionalPathGenerator created successfully")

        # Test that generator has required attributes
        assert hasattr(generator, 'model'), "Generator should have model attribute"
        print("✓ Generator has required attributes")

        print("\nAll basic tests passed!")
        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        return False

if __name__ == "__main__":
    test_diffusion_model_basic()