"""
Tests for memory components of consciousness model.
"""
import pytest
import jax
import jax.numpy as jnp
from tests.unit.test_base import ConsciousnessTestBase
from models.memory import WorkingMemory, InformationIntegration, GRUCell

class TestMemoryComponents(ConsciousnessTestBase):
    """Test suite for memory components."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def seq_length(self):
        return 8

    @pytest.fixture
    def hidden_dim(self):
        return 64

    @pytest.fixture
    def working_memory(self, hidden_dim):
        """Create working memory module for testing."""
        return WorkingMemory(
            hidden_dim=hidden_dim,
            dropout_rate=0.1
        )

    @pytest.fixture
    def info_integration(self, hidden_dim):
        """Create information integration module for testing."""
        return InformationIntegration(
            hidden_dim=hidden_dim,
            num_modules=4,
            dropout_rate=0.1
        )

    @pytest.fixture
    def gru_cell(self, hidden_dim):
        """Create GRU cell for testing."""
        return GRUCell(hidden_dim=hidden_dim)

    def test_gru_state_updates(self, gru_cell, key, batch_size, hidden_dim):
        """Test GRU cell state updates."""
        x = jax.random.normal(key, (batch_size, hidden_dim))
        h = jax.random.normal(key, (batch_size, hidden_dim))

        # Initialize and run forward pass
        variables = gru_cell.init(key, x, h)
        new_h = gru_cell.apply(variables, x, h)

        # Verify shapes
        self.assert_output_shape(new_h, (batch_size, hidden_dim))

        # State should be updated (different from initial state)
        assert not jnp.allclose(new_h, h, rtol=1e-5)

    def test_memory_sequence_processing(self, working_memory, key, batch_size, seq_length, hidden_dim):
        """Test working memory sequence processing."""
        # Test with different sequence lengths
        for test_length in [4, 8, 16]:
            inputs = self.create_inputs(key, batch_size, test_length, hidden_dim)
            initial_state = jnp.zeros((batch_size, hidden_dim))

            input_shape = (batch_size,)
            variables = working_memory.init(
                key, inputs, initial_state=initial_state, deterministic=True
            )
            output, final_state = working_memory.apply(
                variables, inputs, initial_state=initial_state, deterministic=True
            )

            # Verify shapes adapt to sequence length
            self.assert_output_shape(output, (batch_size, test_length, hidden_dim))

    def test_context_aware_gating(self, working_memory, key, batch_size, seq_length, hidden_dim):
        """Test context-aware gating mechanisms."""
        # Create two different input sequences with controlled differences
        base_inputs = self.create_inputs(key, batch_size, seq_length, hidden_dim)

        # Create similar and different inputs
        similar_inputs = base_inputs + jax.random.normal(key, base_inputs.shape) * 0.1
        different_inputs = jax.random.normal(
            jax.random.fold_in(key, 1),
            base_inputs.shape
        )

        initial_state = jnp.zeros((batch_size, hidden_dim))
        input_shape = (batch_size,)
        variables = working_memory.init(
            key, base_inputs, initial_state=initial_state, deterministic=True
        )

        # Process sequences
        _, state_base = working_memory.apply(
            variables, base_inputs, initial_state=initial_state, deterministic=True
        )
        _, state_similar = working_memory.apply(
            variables, similar_inputs, initial_state=initial_state, deterministic=True
        )
        _, state_different = working_memory.apply(
            variables, different_inputs, initial_state=initial_state, deterministic=True
        )

        # Similar inputs should produce more similar states than different inputs
        base_similar_diff = jnp.mean(jnp.abs(state_base - state_similar))
        base_different_diff = jnp.mean(jnp.abs(state_base - state_different))
        assert base_similar_diff < base_different_diff

    def test_information_integration(self, info_integration, key, batch_size, seq_length, hidden_dim):
        """Test information integration computation."""
        # Create inputs with proper shape for information integration
        inputs = jnp.stack([
            self.create_inputs(jax.random.fold_in(key, i), batch_size, seq_length, hidden_dim)
            for i in range(info_integration.num_modules)
        ], axis=1)  # Shape: [batch, num_modules, seq_length, hidden_dim]

        # Initialize and run forward pass
        input_shape = (batch_size,)
        variables = info_integration.init(key, inputs, deterministic=True)
        output, phi = info_integration.apply(variables, inputs, deterministic=True)

        # Verify shapes
        expected_output_shape = (batch_size, info_integration.num_modules, seq_length, hidden_dim)
        self.assert_output_shape(output, expected_output_shape)

        # Phi should be a scalar per batch element
        assert phi.shape == (batch_size,)
        # Phi should be non-negative and finite
        assert jnp.all(jnp.logical_and(phi >= 0, jnp.isfinite(phi)))

    def test_memory_retention(self, working_memory, key, batch_size, seq_length, hidden_dim):
        """Test memory retention over sequences."""
        # Create a sequence with a distinctive pattern
        pattern = jnp.ones((batch_size, 1, hidden_dim))
        inputs = jnp.concatenate([
            pattern,
            self.create_inputs(key, batch_size, seq_length-2, hidden_dim),
            pattern
        ], axis=1)

        initial_state = jnp.zeros((batch_size, hidden_dim))

        input_shape = (batch_size,)
        variables = working_memory.init(
            key, inputs, initial_state=initial_state, deterministic=True
        )
        output, final_state = working_memory.apply(
            variables, inputs, initial_state=initial_state, deterministic=True
        )

        # Final state should capture pattern information
        assert jnp.any(jnp.abs(final_state) > 0.1)  # Non-zero activations
