"""
Unit tests for consciousness attention mechanisms.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest
from jax import random

from models.attention import ConsciousnessAttention, GlobalWorkspace
from models.memory import WorkingMemory

class TestConsciousnessAttention:
    @pytest.fixture
    def key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def attention_module(self):
        return ConsciousnessAttention(
            num_heads=4,
            head_dim=32,
            dropout_rate=0.1
        )

    def test_scaled_dot_product_attention(self, key, attention_module):
        # Test input shapes
        batch_size = 2
        seq_length = 8
        input_dim = 128

        # Create sample inputs
        inputs_q = random.normal(
            key,
            (batch_size, seq_length, input_dim)
        )
        inputs_kv = random.normal(
            key,
            (batch_size, seq_length, input_dim)
        )

        # Initialize parameters
        variables = attention_module.init(key, inputs_q, inputs_kv)

        # Apply attention
        output, attention_weights = attention_module.apply(
            variables,
            inputs_q,
            inputs_kv,
            deterministic=True
        )

        # Test output shapes
        assert output.shape == (batch_size, seq_length, input_dim)
        assert attention_weights.shape == (batch_size, 4, seq_length, seq_length)

        # Test attention weight properties
        # Weights should sum to 1 along the key dimension
        weight_sums = jnp.sum(attention_weights, axis=-1)
        assert jnp.allclose(weight_sums, jnp.ones_like(weight_sums))

        # Test masking
        mask = jnp.ones((batch_size, seq_length), dtype=bool)
        mask = mask.at[:, -1].set(False)  # Mask out last position

        output_masked, attention_weights_masked = attention_module.apply(
            variables,
            inputs_q,
            inputs_kv,
            mask=mask,
            deterministic=True
        )

        # Verify masked positions have zero attention
        assert jnp.allclose(attention_weights_masked[..., -1], 0.0)

    def test_attention_dropout(self, key, attention_module):
        batch_size = 2
        seq_length = 8
        input_dim = 128

        inputs_q = random.normal(key, (batch_size, seq_length, input_dim))
        inputs_kv = random.normal(key, (batch_size, seq_length, input_dim))

        variables = attention_module.init(key, inputs_q, inputs_kv)

        # Test with dropout enabled (training mode)
        output1, _ = attention_module.apply(
            variables,
            inputs_q,
            inputs_kv,
            deterministic=False,
            rngs={'dropout': random.PRNGKey(1)}
        )

        output2, _ = attention_module.apply(
            variables,
            inputs_q,
            inputs_kv,
            deterministic=False,
            rngs={'dropout': random.PRNGKey(2)}
        )

        # Outputs should be different due to dropout
        assert not jnp.allclose(output1, output2)

        # Test with dropout disabled (inference mode)
        output3, _ = attention_module.apply(
            variables,
            inputs_q,
            inputs_kv,
            deterministic=True
        )

        output4, _ = attention_module.apply(
            variables,
            inputs_q,
            inputs_kv,
            deterministic=True
        )

        # Outputs should be identical with dropout disabled
        assert jnp.allclose(output3, output4)

    def test_attention_output_shape(self):
        batch_size = 2
        seq_length = 8
        input_dim = 128

        inputs_q = random.normal(key, (batch_size, seq_length, input_dim))
        inputs_kv = random.normal(key, (batch_size, seq_length, input_dim))

        variables = attention_module.init(key, inputs_q, inputs_kv)

        output, _ = attention_module.apply(
            variables,
            inputs_q,
            inputs_kv,
            deterministic=True
        )

        assert output.shape == inputs_q.shape  # Adjusted expected shape

class TestGlobalWorkspace:
    @pytest.fixture
    def key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def workspace_module(self):
        return GlobalWorkspace(
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            dropout_rate=0.1
        )

    def test_global_workspace_broadcasting(self, key, workspace_module):
        batch_size = 2
        seq_length = 8
        input_dim = 128

        inputs = random.normal(key, (batch_size, seq_length, input_dim))
        variables = workspace_module.init(key, inputs)

        output, attention_weights = workspace_module.apply(
            variables,
            inputs,
            deterministic=True
        )

        # Test output shapes
        assert output.shape == inputs.shape
        assert attention_weights.shape == (batch_size, 4, seq_length, seq_length)

        # Test residual connection
        # Output should not be too different from input due to residual
        assert jnp.mean(jnp.abs(output - inputs)) < 1.2  # Adjust threshold
