"""Test suite for consciousness module implementation."""
import pytest
import jax
import jax.numpy as jnp
from models.consciousness_model import ConsciousnessModel
from tests.unit.test_base import ConsciousnessTestBase

class TestConsciousnessModel(ConsciousnessTestBase):
    """Test cases for the consciousness model."""

    @pytest.fixture
    def model(self, hidden_dim, num_heads):
        """Create a consciousness model for testing."""
        return ConsciousnessModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=4,
            num_states=4,
            dropout_rate=0.1
        )

    @pytest.fixture
    def sample_input(self, key, batch_size, seq_length, hidden_dim):
        """Create sample input data for testing."""
        inputs = {
            'attention': self.create_inputs(key, batch_size, seq_length, hidden_dim),
            'memory': self.create_inputs(jax.random.fold_in(key, 1), batch_size, seq_length, hidden_dim),
            'reasoning': self.create_inputs(jax.random.fold_in(key, 2), batch_size, seq_length, hidden_dim),
            'emotion': self.create_inputs(jax.random.fold_in(key, 3), batch_size, seq_length, hidden_dim)
        }
        return inputs

    @pytest.fixture
    def deterministic(self):
        return True

    def test_model_initialization(self, model):
        """Test that consciousness model initializes correctly."""
        assert isinstance(model, ConsciousnessModel)
        assert model.hidden_dim == 64
        assert model.num_heads == 4
        assert model.num_layers == 4
        assert model.num_states == 4

    def test_model_forward_pass(self, model, sample_input, key, deterministic):
        """Test forward pass through consciousness model."""
        # Initialize model
        variables = model.init(key, sample_input, deterministic=deterministic)

        # Run forward pass
        new_state, metrics = model.apply(
            variables,
            sample_input,
            deterministic=deterministic
        )

        # Check output structure and shapes
        batch_size = sample_input['attention'].shape[0]
        assert new_state.shape == (batch_size, model.hidden_dim)

        # Verify metrics
        assert 'memory_state' in metrics
        assert 'attention_weights' in metrics
        assert 'phi' in metrics
        assert 'attention_maps' in metrics

        # Validate attention weights
        self.assert_valid_attention(metrics['attention_weights'])

    def test_model_config(self, model):
        """Test model configuration methods."""
        config = model.get_config()
        assert config['hidden_dim'] == 64
        assert config['num_heads'] == 4
        assert config['num_layers'] == 4
        assert config['num_states'] == 4
        assert config['dropout_rate'] == 0.1

        default_config = ConsciousnessModel.create_default_config()
        assert isinstance(default_config, dict)
        assert all(k in default_config for k in [
            'hidden_dim', 'num_heads', 'num_layers', 'num_states', 'dropout_rate'
        ])

if __name__ == '__main__':
    pytest.main([__file__])
