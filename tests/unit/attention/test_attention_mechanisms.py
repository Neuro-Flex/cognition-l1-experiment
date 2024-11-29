"""
Comprehensive tests for attention mechanisms in consciousness model.
"""
import pytest
import jax
import jax.numpy as jnp
from tests.unit.test_base import ConsciousnessTestBase
from models.attention import ConsciousnessAttention, GlobalWorkspace

class TestAttentionMechanisms(ConsciousnessTestBase):
    """Test suite for attention mechanisms."""

    @pytest.fixture
    def attention_module(self, hidden_dim, num_heads):
        """Create attention module for testing."""
        return ConsciousnessAttention(
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            dropout_rate=0.1
        )

    def test_scaled_dot_product(self, attention_module, key, batch_size, seq_length, hidden_dim):
        """Test scaled dot-product attention computation."""
        # Create inputs
        inputs_q = self.create_inputs(key, batch_size, seq_length, hidden_dim)
        inputs_kv = self.create_inputs(jax.random.fold_in(key, 1), batch_size, seq_length, hidden_dim)

        # Initialize and run forward pass
        variables = attention_module.init(key, inputs_q, inputs_kv, deterministic=True)
        output, attention_weights = attention_module.apply(
            variables, inputs_q, inputs_kv, deterministic=True
        )

        # Verify output shape
        self.assert_output_shape(output, (batch_size, seq_length, hidden_dim))

        # Verify attention weights
        self.assert_valid_attention(attention_weights)

    def test_attention_mask(self, attention_module, key, batch_size, seq_length, hidden_dim):
        """Test attention mask handling."""
        # Create inputs and mask
        inputs_q = self.create_inputs(key, batch_size, seq_length, hidden_dim)
        inputs_kv = self.create_inputs(jax.random.fold_in(key, 1), batch_size, seq_length, hidden_dim)
        mask = jnp.ones((batch_size, seq_length), dtype=bool)
        mask = mask.at[:, seq_length//2:].set(False)  # Mask out second half

        # Initialize and run forward pass
        variables = attention_module.init(key, inputs_q, inputs_kv, mask=mask, deterministic=True)
        output, attention_weights = attention_module.apply(
            variables, inputs_q, inputs_kv, mask=mask, deterministic=True
        )

        # Verify masked attention weights are zero
        assert jnp.allclose(attention_weights[..., seq_length//2:], 0.0)

    def test_consciousness_broadcasting(self, attention_module, key, batch_size, seq_length, hidden_dim):
        """Test consciousness-aware broadcasting."""
        inputs_q = self.create_inputs(key, batch_size, seq_length, hidden_dim)
        inputs_kv = self.create_inputs(jax.random.fold_in(key, 1), batch_size, seq_length, hidden_dim)

        # Test with and without dropout
        variables = attention_module.init(key, inputs_q, inputs_kv, deterministic=True)

        # Test deterministic output
        output1, _ = attention_module.apply(
            variables, inputs_q, inputs_kv, deterministic=True
        )
        output2, _ = attention_module.apply(
            variables, inputs_q, inputs_kv, deterministic=True
        )

        # Outputs should be identical when deterministic
        assert jnp.allclose(output1, output2, rtol=1e-5)

    def test_global_workspace_integration(self, key, batch_size, seq_length, hidden_dim, num_heads):
        """Test global workspace integration."""
        workspace = GlobalWorkspace(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            dropout_rate=0.1
        )

        inputs = self.create_inputs(key, batch_size, seq_length, hidden_dim)

        # Initialize and run forward pass
        variables = workspace.init(key, inputs, deterministic=True)
        output, attention_weights = workspace.apply(
            variables, inputs, deterministic=True
        )

        # Verify shapes
        self.assert_output_shape(output, (batch_size, seq_length, hidden_dim))

        # Test residual connection
        # Output should be different from input due to processing
        assert not jnp.allclose(output, inputs, rtol=1e-5)
