"""Comprehensive environment testing for AI consciousness implementation."""
import sys
import unittest
import platform

class EnvironmentTests(unittest.TestCase):
    """Test suite for verifying environment setup and dependencies."""

    def test_python_version(self):
        """Verify Python version is 3.8+"""
        major, minor = sys.version_info[:2]
        self.assertGreaterEqual(major, 3)
        self.assertGreaterEqual(minor, 8)

    def test_core_imports(self):
        """Test all core framework imports"""
        try:
            import jax
            import jax.numpy as jnp
            import flax
            import optax
            import torch
            self.assertTrue(True, "All core imports successful")
        except ImportError as e:
            self.fail(f"Failed to import core frameworks: {str(e)}")

    def test_hardware_detection(self):
        """Test hardware detection and configuration"""
        import jax
        import torch

        # Check JAX devices
        devices = jax.devices()
        self.assertGreater(len(devices), 0, "No JAX devices found")
        print(f"JAX devices: {devices}")

        # Check PyTorch devices
        self.assertTrue(hasattr(torch, 'cuda'))
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    def test_memory_allocation(self):
        """Test basic memory operations"""
        import jax.numpy as jnp
        import torch

        try:
            # Test JAX array creation
            x = jnp.ones((1000, 1000))
            self.assertEqual(x.shape, (1000, 1000))

            # Test PyTorch tensor creation
            y = torch.ones(1000, 1000)
            self.assertEqual(y.shape, (1000, 1000))
        except Exception as e:
            self.fail(f"Memory allocation test failed: {str(e)}")

    def test_framework_versions(self):
        """Verify framework versions"""
        import jax
        import flax
        import optax
        import torch

        versions = {
            'jax': jax.__version__,
            'flax': flax.__version__,
            'optax': optax.__version__,
            'torch': torch.__version__
        }

        print("\nFramework versions:")
        for framework, version in versions.items():
            print(f"{framework}: {version}")
            self.assertIsNotNone(version)

if __name__ == '__main__':
    print(f"Running environment tests on Python {sys.version}")
    print(f"Platform: {platform.platform()}")
    unittest.main(verbosity=2)
