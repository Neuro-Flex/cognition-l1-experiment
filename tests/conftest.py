"""
Pytest configuration and shared fixtures.
"""
import pytest
import jax
import jax.numpy as jnp
from jax import random

# Configure JAX for CPU optimization
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)  # Use float32 for memory efficiency

# Test configuration for CPU optimization
BATCH_SIZE = 4
SEQ_LENGTH = 16
HIDDEN_DIM = 32

@pytest.fixture(scope="session")
def global_key():
    """Provide a global random key for reproducible tests."""
    return random.PRNGKey(42)

@pytest.fixture(scope="session")
def device():
    """Ensure tests run on CPU for consistent behavior."""
    return jax.devices("cpu")[0]

@pytest.fixture
def batch_size():
    """Return optimized batch size for tests."""
    return BATCH_SIZE

@pytest.fixture
def seq_length():
    """Return optimized sequence length for tests."""
    return SEQ_LENGTH

@pytest.fixture
def hidden_dim():
    """Return optimized hidden dimension for tests."""
    return HIDDEN_DIM

@pytest.fixture(autouse=True)
def jit_compile():
    """Apply JIT compilation to test functions."""
    def decorator(func):
        return jax.jit(func, static_argnums=(0,))
    return decorator
