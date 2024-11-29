"""
Unit tests for cognitive process integration components.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest
from jax import random

from models.consciousness_state import CognitiveProcessIntegration

class TestCognitiveProcessIntegration:
    @pytest.fixture
    def key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def integration_module(self):
        return CognitiveProcessIntegration(
            hidden_dim=64,
            num_heads=4,
            num_layers=3,
            dropout_rate=0.1
        )

    def test_cross_modal_attention(self, key, integration_module):
        # Test dimensions
        batch_size = 2
        seq_length = 8
        input_dim = 32

        # Create multi-modal inputs
        inputs = {
            'visual': random.normal(key, (batch_size, seq_length, input_dim)),
            'textual': random.normal(
                random.PRNGKey(1),
                (batch_size, seq_length, input_dim)
            ),
            'numerical': random.normal(
                random.PRNGKey(2),
                (batch_size, seq_length, input_dim)
            )
        }

        # Initialize parameters
        variables = integration_module.init(key, inputs)

        # Process through integration
        consciousness_state, attention_maps = integration_module.apply(
            variables,
            inputs,
            deterministic=True
        )

        # Test output shapes
        assert consciousness_state.shape == (batch_size, seq_length, 64)

        # Test attention maps
        for source in inputs.keys():
            for target in inputs.keys():
                if source != target:
                    map_key = f"{target}-{source}"
                    assert map_key in attention_maps
                    attention_map = attention_maps[map_key]
                    # Check attention map properties
                    assert attention_map.shape[-2:] == (seq_length, seq_length)
                    # Verify attention weights sum to 1
                    assert jnp.allclose(
                        jnp.sum(attention_map, axis=-1),
                        jnp.ones((batch_size, 4, seq_length))
                    )

    def test_modality_specific_processing(self, key, integration_module):
        batch_size = 2
        seq_length = 8
        input_dim = 32

        # Test with single modality
        single_input = {
            'visual': random.normal(key, (batch_size, seq_length, input_dim))
        }
        variables = integration_module.init(key, single_input)

        consciousness_state1, _ = integration_module.apply(
            variables,
            single_input,
            deterministic=True
        )

        # Test with multiple modalities
        multi_input = {
            'visual': single_input['visual'],
            'textual': random.normal(key, (batch_size, seq_length, input_dim))
        }

        consciousness_state2, _ = integration_module.apply(
            variables,
            multi_input,
            deterministic=True
        )

        # Multi-modal processing should produce different results
        assert not jnp.allclose(consciousness_state1, consciousness_state2)

    def test_integration_stability(self, key, integration_module):
        batch_size = 2
        seq_length = 8
        input_dim = 32

        inputs = {
            'modality1': random.normal(key, (batch_size, seq_length, input_dim)),
            'modality2': random.normal(key, (batch_size, seq_length, input_dim))
        }

        variables = integration_module.init(key, inputs)

        # Test stability across multiple forward passes
        states = []
        for _ in range(5):
            state, _ = integration_module.apply(
                variables,
                inputs,
                deterministic=True
            )
            states.append(state)

        # All forward passes should produce identical results
        for i in range(1, len(states)):
            assert jnp.allclose(states[0], states[i])

        # Test with dropout
        states_dropout = []
        for i in range(5):
            state, _ = integration_module.apply(
                variables,
                inputs,
                deterministic=False,
                rngs={'dropout': random.PRNGKey(i)}
            )
            states_dropout.append(state)

        # Dropout should produce different results
        assert not all(
            jnp.allclose(states_dropout[0], state)
            for state in states_dropout[1:]
        )
