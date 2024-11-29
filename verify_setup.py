"""
Verification script for AI consciousness development environment.
"""
import os
import sys
from typing import Dict, List, Any

def verify_imports() -> Dict[str, bool]:
    """Verify all required packages are properly installed."""
    results = {}
    packages = {
        'jax': ['jax', 'jax.numpy'],
        'flax': ['flax', 'flax.linen'],
        'optax': ['optax'],
        'torch': ['torch'],
        'transformers': ['transformers']
    }

    for package, modules in packages.items():
        try:
            for module in modules:
                __import__(module)
            results[package] = True
        except ImportError as e:
            results[package] = False
            print(f"Failed to import {package}: {str(e)}")
    return results

def verify_hardware_detection() -> Dict[str, Any]:
    """Verify hardware detection and configuration."""
    hardware_info = {}

    # Check JAX devices
    try:
        import jax
        hardware_info['jax_devices'] = str(jax.devices())
    except Exception as e:
        hardware_info['jax_devices'] = f"Error: {str(e)}"

    # Check PyTorch devices
    try:
        import torch
        hardware_info['torch_cuda'] = torch.cuda.is_available()
        hardware_info['torch_device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        hardware_info['torch_cuda'] = f"Error: {str(e)}"

    return hardware_info

def verify_basic_operations() -> Dict[str, bool]:
    """Verify basic ML operations work."""
    results = {}

    # Test JAX operations
    try:
        import jax.numpy as jnp
        x = jnp.ones((100, 100))
        y = jnp.dot(x, x)
        results['jax_operations'] = True
    except Exception as e:
        results['jax_operations'] = False
        print(f"JAX operations failed: {str(e)}")

    # Test PyTorch operations
    try:
        import torch
        x = torch.ones(100, 100)
        y = torch.matmul(x, x)
        results['torch_operations'] = True
    except Exception as e:
        results['torch_operations'] = False
        print(f"PyTorch operations failed: {str(e)}")

    return results

def main():
    """Run all verification checks."""
    print("Verifying AI consciousness development environment...\n")

    # Check Python version
    print(f"Python version: {sys.version}")

    # Verify imports
    print("\nVerifying package imports...")
    import_results = verify_imports()
    for package, success in import_results.items():
        print(f"{package}: {'✓' if success else '✗'}")

    # Verify hardware detection
    print("\nVerifying hardware detection...")
    hardware_info = verify_hardware_detection()
    for key, value in hardware_info.items():
        print(f"{key}: {value}")

    # Verify basic operations
    print("\nVerifying basic operations...")
    operation_results = verify_basic_operations()
    for operation, success in operation_results.items():
        print(f"{operation}: {'✓' if success else '✗'}")

    # Overall status
    all_successful = (
        all(import_results.values()) and
        all(v != False for v in hardware_info.values()) and
        all(operation_results.values())
    )

    print("\nOverall Status:", "✓ Ready" if all_successful else "✗ Issues detected")

    return 0 if all_successful else 1

if __name__ == '__main__':
    sys.exit(main())
