"""
Unit tests for working memory and GRU components.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest
from jax import random

from models.memory import GRUCell, WorkingMemory, InformationIntegration

class TestGRUCell:
    @pytest.fixture
    def key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def gru_cell(self):
        return GRUCell(hidden_dim=64)

    def test_gru_state_updates(self, key, gru_cell):
        # Test dimensions
        batch_size = 2
        input_dim = 32
        hidden_dim = 64

        # Create sample inputs
        x = random.normal(key, (batch_size, input_dim))
        h = random.normal(key, (batch_size, hidden_dim))

        # Initialize parameters
        variables = gru_cell.init(key, x, h)

        # Apply GRU cell
        new_h = gru_cell.apply(variables, x, h)

        # Test output shape
        assert new_h.shape == (batch_size, hidden_dim)

        # Test state update properties
        # Values should be bounded by tanh activation
        assert jnp.all(jnp.abs(new_h) <= 1.0)

        # Test multiple updates maintain reasonable values
        for _ in range(10):
            h = new_h
            new_h = gru_cell.apply(variables, x, h)
            assert jnp.all(jnp.isfinite(new_h))
            assert jnp.all(jnp.abs(new_h) <= 1.0)

    def test_gru_reset_gate(self, key, gru_cell):
        batch_size = 2
        input_dim = 32
        hidden_dim = 64

        x = random.normal(key, (batch_size, input_dim))
        h = random.normal(key, (batch_size, hidden_dim))

        variables = gru_cell.init(key, x, h)

        # Test with zero input
        x_zero = jnp.zeros_like(x)
        h_zero = gru_cell.apply(variables, x_zero, h)

        # With zero input, new state should be influenced by reset gate
        # and should be different from previous state
        assert not jnp.allclose(h_zero, h)

        # Test with zero state
        h_zero = jnp.zeros_like(h)
        new_h = gru_cell.apply(variables, x, h_zero)

        # With zero state, output should be primarily determined by input
        assert not jnp.allclose(new_h, h_zero)

class TestWorkingMemory:
    @pytest.fixture
    def key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def memory_module(self):
        return WorkingMemory(hidden_dim=64, dropout_rate=0.1)

    def test_sequence_processing(self, key, memory_module):
        batch_size = 2
        seq_length = 8
        input_dim = 32
        hidden_dim = 64

        # Create sample sequence
        inputs = random.normal(key, (batch_size, seq_length, input_dim))
        
        # Initialize parameters
        input_shape = (hidden_dim,)
        variables = memory_module.init(key, inputs, deterministic=True)
        
        # Process sequence
        outputs, final_state = memory_module.apply(
            variables,
            inputs,
            deterministic=True
        )

        # Test output shapes
        assert outputs.shape == (batch_size, seq_length, hidden_dim)
        assert final_state.shape == (batch_size, hidden_dim)

        # Test temporal consistency
        # Later timesteps should be influenced by earlier ones
        first_half = outputs[:, :seq_length//2, :]
        second_half = outputs[:, seq_length//2:, :]

        # Calculate temporal correlation
        correlation = jnp.mean(jnp.abs(
            jnp.corrcoef(
                first_half.reshape(-1, hidden_dim),
                second_half.reshape(-1, hidden_dim)
            )
        ))

        # There should be some correlation between timesteps
        assert correlation > 0.1

    def test_memory_retention(self, key, memory_module):
        batch_size = 2
        seq_length = 8
        input_dim = 32

        inputs = random.normal(key, (batch_size, seq_length, input_dim))
        input_shape = (batch_size,)
        variables = memory_module.init(key, inputs)

        # Test with different initial states
        initial_state = random.normal(key, (batch_size, 64))

        outputs1, final_state1 = memory_module.apply(
            variables,
            inputs,
            initial_state=initial_state,
            deterministic=True
        )

        outputs2, final_state2 = memory_module.apply(
            variables,
            inputs,
            initial_state=jnp.zeros_like(initial_state),
            deterministic=True
        )

        # Different initial states should lead to different outputs
        assert not jnp.allclose(outputs1, outputs2)
        assert not jnp.allclose(final_state1, final_state2)
