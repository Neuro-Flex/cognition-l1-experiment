import jax
import jax.numpy as jnp
import pytest
from jax import random
from typing import Dict, Tuple
from flax.training import train_state
import optax

from models.consciousness_model import ConsciousnessModel

class TestARCReasoning:
    @pytest.fixture
    def key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def model_config(self):
        return ConsciousnessModel.create_default_config()

    @pytest.fixture
    def consciousness_model(self, model_config):
        return ConsciousnessModel(**model_config)

    def load_arc_sample(self) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """Load a sample ARC task for testing."""
        # Sample pattern with proper shape (batch, height, width, channels)
        sample_input = {
            'visual': jnp.array([
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ], dtype=jnp.float32)[None, :, :, None]
        }

        expected_output = jnp.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=jnp.float32)[None, :, :, None]

        return sample_input, expected_output

    def test_pattern_recognition(self, key, consciousness_model):
        inputs, expected = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]
        
        # Initialize model state
        model_inputs = {
            'visual': inputs['visual'],
            'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))
        }

        # Ensure input_shape is a tuple
        input_shape = (consciousness_model.hidden_dim,)
        # Initialize model
        variables = consciousness_model.init(key, model_inputs)

        try:
            # Forward pass
            output, metrics = consciousness_model.apply(
                variables,
                model_inputs,
                deterministic=True,
                consciousness_threshold=0.5
            )

            # Validate outputs
            assert output.shape == (batch_size, consciousness_model.hidden_dim)
            assert 'phi' in metrics
            assert metrics['phi'].shape == (batch_size,)
            assert jnp.all(metrics['phi'] >= 0)

            # Validate attention
            assert 'attention_weights' in metrics
            assert metrics['attention_weights'].ndim >= 3  # (batch, heads, seq)

            # Validate attention maps
            assert 'attention_maps' in metrics
            for attn_map in metrics['attention_maps'].values():
                assert jnp.allclose(
                    jnp.sum(attn_map, axis=-1),
                    jnp.ones((batch_size, 8, 64))  # (batch, heads, seq_length)
                )

        except Exception as e:
            pytest.fail(f"Pattern recognition test failed: {str(e)}")

    def test_abstraction_capability(self, key, consciousness_model):
        inputs, _ = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]

        # Create transformed versions
        variations = {
            'original': inputs['visual'],
            'rotated': jnp.rot90(inputs['visual'][:, :, :, 0], k=1)[:, :, None],
            'scaled': inputs['visual'] * 2.0
        }

        try:
            # Ensure input_shape is a tuple
            input_shape = (consciousness_model.hidden_dim,)
            variables = consciousness_model.init(
                key, 
                {'visual': variations['original'], 
                 'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))}
            )

            states = {}
            for name, visual_input in variations.items():
                output, metrics = consciousness_model.apply(
                    variables,
                    {'visual': visual_input, 
                     'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))},
                    deterministic=True
                )
                states[name] = output

            # Test representation similarity
            def cosine_similarity(x, y):
                return jnp.sum(x * y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))

            orig_rot_sim = cosine_similarity(
                states['original'].ravel(),
                states['rotated'].ravel()
            )
            orig_scaled_sim = cosine_similarity(
                states['original'].ravel(),
                states['scaled'].ravel()
            )

            # Transformed versions should maintain similar representations
            assert orig_rot_sim > 0.5
            assert orig_scaled_sim > 0.7

        except Exception as e:
            pytest.fail(f"Abstraction capability test failed: {str(e)}")

    def test_conscious_adaptation(self, key, consciousness_model):
        inputs, _ = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]

        try:
            # Create simple and complex patterns
            simple_input = {
                'visual': inputs['visual'],
                'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))
            }
            
            # More complex pattern (doubled size)
            complex_visual = jnp.tile(inputs['visual'], (1, 2, 2, 1))
            complex_input = {
                'visual': complex_visual,
                'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))
            }

            # Ensure input_shape is a tuple
            input_shape = (consciousness_model.hidden_dim,)
            variables = consciousness_model.init(key, simple_input)

            # Process both patterns
            _, simple_metrics = consciousness_model.apply(
                variables,
                simple_input,
                deterministic=True
            )
            
            _, complex_metrics = consciousness_model.apply(
                variables,
                complex_input,
                deterministic=True
            )

            # Validate complexity adaptation
            assert complex_metrics['phi'] > simple_metrics['phi']
            assert 'attention_weights' in simple_metrics
            assert 'attention_weights' in complex_metrics

            # Validate attention maps
            assert 'attention_maps' in simple_metrics
            assert 'attention_maps' in complex_metrics
            for attn_map in simple_metrics['attention_maps'].values():
                assert jnp.allclose(
                    jnp.sum(attn_map, axis=-1),
                    jnp.ones((batch_size, 8, 64))  # (batch, heads, seq_length)
                )
            for attn_map in complex_metrics['attention_maps'].values():
                assert jnp.allclose(
                    jnp.sum(attn_map, axis=-1),
                    jnp.ones((batch_size, 8, 64))  # (batch, heads, seq_length)
                )

        except Exception as e:
            pytest.fail(f"Conscious adaptation test failed: {str(e)}")

    def test_working_memory(self, key, consciousness_model):
        inputs, _ = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]

        # Initialize model state
        model_inputs = {
            'visual': inputs['visual'],
            'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))
        }

        # Ensure input_shape is a tuple
        input_shape = (consciousness_model.hidden_dim,)
        variables = consciousness_model.init(key, model_inputs)

        try:
            # Forward pass
            output, metrics = consciousness_model.apply(
                variables,
                model_inputs,
                deterministic=True,
                consciousness_threshold=0.5
            )

            # Validate working memory outputs
            assert 'memory_state' in metrics
            assert metrics['memory_state'].shape == (batch_size, consciousness_model.hidden_dim)

        except Exception as e:
            pytest.fail(f"Working memory test failed: {str(e)}")

    def test_cognitive_process_integration(self, key, consciousness_model):
        inputs, _ = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]

        # Initialize model state
        model_inputs = {
            'visual': inputs['visual'],
            'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))
        }

        # Ensure input_shape is a tuple
        input_shape = (consciousness_model.hidden_dim,)
        variables = consciousness_model.init(key, model_inputs)

        try:
            # Forward pass
            output, metrics = consciousness_model.apply(
                variables,
                model_inputs,
                deterministic=True,
                consciousness_threshold=0.5
            )

            # Validate cognitive process integration outputs
            assert 'attention_maps' in metrics
            for attn_map in metrics['attention_maps'].values():
                assert jnp.allclose(
                    jnp.sum(attn_map, axis=-1),
                    jnp.ones((batch_size, 8, 64))  # (batch, heads, seq_length)
                )

        except Exception as e:
            pytest.fail(f"Cognitive process integration test failed: {str(e)}")

    def test_consciousness_state_manager(self, key, consciousness_model):
        inputs, _ = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]

        # Initialize model state
        model_inputs = {
            'visual': inputs['visual'],
            'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))
        }

        # Ensure input_shape is a tuple
        input_shape = (consciousness_model.hidden_dim,)
        variables = consciousness_model.init(key, model_inputs)

        try:
            # Forward pass
            output, metrics = consciousness_model.apply(
                variables,
                model_inputs,
                deterministic=True,
                consciousness_threshold=0.5
            )

            # Validate consciousness state manager outputs
            assert 'memory_gate' in metrics
            assert metrics['memory_gate'].shape == (batch_size, consciousness_model.hidden_dim)
            assert jnp.all(metrics['memory_gate'] >= 0.0)
            assert jnp.all(metrics['memory_gate'] <= 1.0)

        except Exception as e:
            pytest.fail(f"Consciousness state manager test failed: {str(e)}")

    def test_information_integration(self, key, consciousness_model):
        inputs, _ = self.load_arc_sample()
        batch_size = inputs['visual'].shape[0]

        # Initialize model state
        model_inputs = {
            'visual': inputs['visual'],
            'state': jnp.zeros((batch_size, consciousness_model.hidden_dim))
        }

        # Ensure input_shape is a tuple
        input_shape = (consciousness_model.hidden_dim,)
        variables = consciousness_model.init(key, model_inputs)

        try:
            # Forward pass
            output, metrics = consciousness_model.apply(
                variables,
                model_inputs,
                deterministic=True,
                consciousness_threshold=0.5
            )

            # Validate information integration outputs
            assert 'phi' in metrics
            assert metrics['phi'].shape == (batch_size,)
            assert jnp.all(metrics['phi'] >= 0)

        except Exception as e:
            pytest.fail(f"Information integration test failed: {str(e)}")
