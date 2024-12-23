"""
Unit tests for Information Integration Theory (IIT) components.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest
from jax import random

from models.memory import InformationIntegration

class TestInformationIntegration:
    @pytest.fixture
    def key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def integration_module(self):
        return InformationIntegration(
            hidden_dim=64,
            num_modules=4,
            dropout_rate=0.1
        )

    def test_phi_metric_computation(self, key, integration_module):
        # Test dimensions
        batch_size = 2
        num_modules = 4
        input_dim = 32

        # Create sample inputs
        inputs = random.normal(key, (batch_size, num_modules, input_dim))

        # Initialize parameters
        input_shape = (integration_module.hidden_dim,)
        variables = integration_module.init(key, inputs)

        # Process through integration
        output, phi = integration_module.apply(
            variables,
            inputs,
            deterministic=True
        )

        # Test output shapes
        assert output.shape == (batch_size, num_modules, integration_module.hidden_dim)

        # Test phi properties
        assert jnp.all(jnp.isfinite(phi))  # Phi should be finite
        assert jnp.all(phi >= 0.0)  # Phi should be non-negative

        # Test with different input patterns
        # More structured input should lead to higher phi
        structured_input = jnp.tile(
            random.normal(key, (batch_size, 1, input_dim)),
            (1, num_modules, 1)
        )
        _, phi_structured = integration_module.apply(
            variables,
            structured_input,
            deterministic=True
        )

        random_input = random.normal(key, (batch_size, num_modules, input_dim))
        _, phi_random = integration_module.apply(
            variables,
            random_input,
            deterministic=True
        )

        # Structured input should have higher integration
        assert jnp.all(phi_structured > phi_random)

    def test_information_flow(self, key, integration_module):
        batch_size = 2
        num_modules = 4
        input_dim = 32

        inputs = jnp.zeros((2, 4, 64), dtype=jnp.float32)  # ensure shape matches the model
        input_shape = (integration_module.hidden_dim,)
        variables = integration_module.init(key, inputs)

        # Test with and without dropout
        output1, _ = integration_module.apply(
            variables,
            inputs,
            deterministic=False,
            rngs={'dropout': random.PRNGKey(1)}
        )

        output2, _ = integration_module.apply(
            variables,
            inputs,
            deterministic=True
        )

        # Test residual connection properties
        # Output should maintain some similarity with input
        input_output_correlation = jnp.mean(jnp.abs(
            jnp.corrcoef(
                inputs.reshape(-1, input_dim),
                output2.reshape(-1, input_dim)
            )
        ))
        assert input_output_correlation > 0.1

        # Test module interactions
        # Compute cross-module correlations
        module_correlations = jnp.corrcoef(
            output2.reshape(batch_size * num_modules, input_dim)
        )

        # There should be some correlation between modules
        avg_cross_correlation = jnp.mean(jnp.abs(module_correlations))
        assert avg_cross_correlation > 0.1

    def test_entropy_calculations(self, key, integration_module):
        batch_size = 2
        num_modules = 4
        input_dim = 32

        # Test with different input distributions
        # Uniform distribution
        uniform_input = jnp.ones((batch_size, num_modules, input_dim))
        variables = integration_module.init(key, uniform_input)

        _, phi_uniform = integration_module.apply(
            variables,
            uniform_input,
            deterministic=True
        )

        # Concentrated distribution
        concentrated_input = jnp.zeros((batch_size, num_modules, input_dim))
        concentrated_input = concentrated_input.at[:, :, 0].set(1.0)
        _, phi_concentrated = integration_module.apply(
            variables,
            concentrated_input,
            deterministic=True
        )

        # Uniform distribution should have higher entropy
        assert jnp.all(phi_uniform > phi_concentrated)

    def test_memory_integration(self, key, integration_module):
        batch_size = 2
        num_modules = 4
        input_dim = 32

        inputs = random.normal(key, (batch_size, num_modules, input_dim))
        input_shape = (integration_module.hidden_dim,)
        variables = integration_module.init(key, inputs)

        # Process through integration
        output, phi = integration_module.apply(
            variables,
            inputs,
            deterministic=True
        )

        # Test output shapes
        assert output.shape == (batch_size, num_modules, integration_module.hidden_dim)
        assert phi.shape == (batch_size,)  # Phi should be a scalar per batch element

        # Test phi properties
        assert jnp.all(jnp.isfinite(phi))  # Phi should be finite
        assert jnp.all(phi >= 0.0)  # Phi should be non-negative

        # Test with different input patterns
        # More structured input should lead to higher phi
        structured_input = jnp.tile(
            random.normal(key, (batch_size, 1, input_dim)),
            (1, num_modules, 1)
        )
        _, phi_structured = integration_module.apply(
            variables,
            structured_input,
            deterministic=True
        )

        random_input = random.normal(key, (batch_size, num_modules, input_dim))
        _, phi_random = integration_module.apply(
            variables,
            random_input,
            deterministic=True
        )

        # Structured input should have higher integration
        assert jnp.all(phi_structured > phi_random)
