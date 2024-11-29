"""
Benchmark tests using ARC (Abstract Reasoning Corpus) for consciousness model evaluation.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest
from jax import random
import json
import os
from typing import Dict, List, Tuple

from models.consciousness_model import ConsciousnessModel

class TestARCReasoning:
    @pytest.fixture
    def key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def consciousness_model(self):
        return ConsciousnessModel(
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            num_states=4,
            dropout_rate=0.1
        )

    def load_arc_sample(self) -> Tuple[Dict, Dict]:
        """
        Load a sample ARC task for testing.
        Returns simplified version of task for testing.
        """
        # Sample ARC task: Pattern completion
        # Input: 3x3 grid with a simple pattern
        # Output: Expected completion of pattern
        sample_input = {
            'visual': jnp.array([
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 0]  # Last element missing from pattern
            ], dtype=jnp.float32).reshape(1, 9, 1)
        }

        expected_output = jnp.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]  # Complete pattern
        ], dtype=jnp.float32).reshape(1, 9, 1)

        return sample_input, expected_output

    def test_pattern_recognition(self, key, consciousness_model):
        # Load sample ARC task
        inputs, expected = self.load_arc_sample()

        # Initialize model
        variables = consciousness_model.init(key, inputs)

        # Process through consciousness model
        consciousness_state, metrics = consciousness_model.apply(
            variables,
            inputs,
            deterministic=True
        )

        # Evaluate pattern recognition
        # Convert consciousness state to pattern prediction
        prediction = nn.Dense(1)(consciousness_state)
        prediction = jnp.reshape(prediction, expected.shape)

        # Calculate accuracy
        accuracy = jnp.mean(jnp.abs(prediction - expected) < 0.5)

        # Test metrics
        assert 'phi' in metrics  # Information integration metric
        assert 'attention_maps' in metrics  # Attention patterns
        assert 'memory_state' in metrics  # Working memory state

        # Basic performance checks
        assert accuracy > 0.5  # Should be better than random
        assert metrics['phi'] > 0  # Should show information integration

    def test_abstraction_capability(self, key, consciousness_model):
        # Test ability to abstract patterns across different representations
        inputs, _ = self.load_arc_sample()

        # Create variations of the same pattern
        variations = {
            'original': inputs['visual'],
            'rotated': jnp.rot90(inputs['visual'].reshape(3, 3)).reshape(1, 9, 1),
            'scaled': inputs['visual'] * 2
        }

        variables = consciousness_model.init(key, {'visual': variations['original']})

        # Process each variation
        consciousness_states = {}
        for name, input_var in variations.items():
            state, _ = consciousness_model.apply(
                variables,
                {'visual': input_var},
                deterministic=True
            )
            consciousness_states[name] = state

        # Compare representations
        # Similar patterns should have similar consciousness states
        def state_similarity(state1, state2):
            return jnp.mean(jnp.abs(state1 - state2))

        # Test invariance properties
        original_rotated_sim = state_similarity(
            consciousness_states['original'],
            consciousness_states['rotated']
        )
        original_scaled_sim = state_similarity(
            consciousness_states['original'],
            consciousness_states['scaled']
        )

        # Representations should show some similarity despite transformations
        assert original_rotated_sim < 0.5
        assert original_scaled_sim < 0.5

    def test_conscious_adaptation(self, key, consciousness_model):
        inputs, _ = self.load_arc_sample()
        variables = consciousness_model.init(key, inputs)

        # Test adaptation to pattern difficulty
        simple_pattern = inputs
        complex_pattern = {
            'visual': jnp.tile(inputs['visual'], (1, 2, 1))  # More complex pattern
        }

        # Process patterns
        _, metrics_simple = consciousness_model.apply(
            variables,
            simple_pattern,
            deterministic=True
        )

        _, metrics_complex = consciousness_model.apply(
            variables,
            complex_pattern,
            deterministic=True
        )

        # Complex patterns should show higher integration
        assert metrics_complex['phi'] > metrics_simple['phi']

        # Check energy efficiency adaptation
        assert metrics_complex['energy_cost'] > metrics_simple['energy_cost']
