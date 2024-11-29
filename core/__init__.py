"""
Core module for AI consciousness implementation using JAX, Flax, and Optax.
Provides base implementations for neural network architectures and training utilities.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# Configure JAX to use 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

# Check available devices and optimize accordingly
DEVICES = jax.devices()
print(f"Available devices: {DEVICES}")

__version__ = "0.1.0"
