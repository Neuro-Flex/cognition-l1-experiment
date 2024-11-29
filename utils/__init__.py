"""
Utility functions for AI consciousness implementation.
Provides hardware optimization, logging, and monitoring tools.
"""

import os
import logging
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
import torch

def setup_hardware_environment(use_gpu: bool = True, memory_limit: Optional[float] = None) -> Dict[str, Any]:
    """
    Configure hardware environment for optimal performance.

    Args:
        use_gpu: Whether to enable GPU acceleration if available
        memory_limit: Optional memory limit in GB for GPU memory allocation

    Returns:
        Dict containing hardware configuration details
    """
    # Configure JAX memory allocation
    if memory_limit is not None and use_gpu:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_limit)

    # Get device information
    devices = jax.devices()
    device_type = devices[0].device_kind

    # Configure PyTorch device
    torch_device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    return {
        "jax_devices": devices,
        "jax_platform": jax.default_backend(),
        "torch_device": torch_device,
        "gpu_available": torch.cuda.is_available() or device_type == "gpu",
        "device_count": len(devices)
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__version__ = "0.1.0"
