"""
Tests for consciousness state management.
"""
import pytest
import jax
import jax.numpy as jnp
from tests.unit.test_base import ConsciousnessTestBase
from models.consciousness_state import ConsciousnessStateManager

class TestStateManagement(ConsciousnessTestBase):
    """Test suite for consciousness state management."""

    @pytest.fixture
    def state_manager(self, hidden_dim):
        """Create state management module for testing."""
        return ConsciousnessStateManager(
            hidden_dim=hidden_dim,
            num_states=4,
            dropout_rate=0.1
        )

    def test_state_updates(self, state_manager, key, batch_size, hidden_dim):
        """Test consciousness state updates."""
        consciousness_state = self.create_inputs(key, batch_size, 1, hidden_dim).squeeze(1)
        integrated_output = self.create_inputs(
            jax.random.fold_in(key, 1), batch_size, 1, hidden_dim
        ).squeeze(1)

        # Initialize and run forward pass
        variables = state_manager.init(
            key,
            consciousness_state,
            integrated_output,
            deterministic=True
        )
        new_state, metrics = state_manager.apply(
            variables,
            consciousness_state,
            integrated_output,
            deterministic=True
        )

        # Verify shapes
        self.assert_output_shape(new_state, (batch_size, hidden_dim))
        assert 'state_value' in metrics
        assert 'energy_cost' in metrics

    def test_rl_optimization(self, state_manager, key, batch_size, hidden_dim):
        """Test reinforcement learning optimization."""
        consciousness_state = self.create_inputs(key, batch_size, 1, hidden_dim).squeeze(1)
        integrated_output = self.create_inputs(
            jax.random.fold_in(key, 1), batch_size, 1, hidden_dim
        ).squeeze(1)

        variables = state_manager.init(
            key,
            consciousness_state,
            integrated_output,
            deterministic=True
        )

        # Run multiple updates
        states = []
        values = []
        for _ in range(3):
            new_state, metrics = state_manager.apply(
                variables,
                consciousness_state,
                integrated_output,
                deterministic=True
            )
            states.append(new_state)
            values.append(metrics['state_value'])
            consciousness_state = new_state

        # Check state evolution
        states = jnp.stack(states)
        values = jnp.stack(values)

        # States should change over time
        assert not jnp.allclose(states[0], states[-1], rtol=1e-5)

    def test_energy_efficiency(self, state_manager, key, batch_size, hidden_dim):
        """Test energy efficiency metrics."""
        # Test with different complexity inputs
        simple_state = jnp.zeros((batch_size, hidden_dim))
        complex_state = self.create_inputs(key, batch_size, 1, hidden_dim).squeeze(1)
        integrated_output = self.create_inputs(
            jax.random.fold_in(key, 1), batch_size, 1, hidden_dim
        ).squeeze(1)

        variables = state_manager.init(
            key,
            simple_state,
            integrated_output,
            deterministic=True
        )

        # Compare energy costs
        _, metrics_simple = state_manager.apply(
            variables,
            simple_state,
            integrated_output,
            deterministic=True
        )
        _, metrics_complex = state_manager.apply(
            variables,
            complex_state,
            integrated_output,
            deterministic=True
        )

        # Complex states should require more energy
        assert metrics_complex['energy_cost'] > metrics_simple['energy_cost']

    def test_state_value_estimation(self, state_manager, key, batch_size, hidden_dim):
        """Test state value estimation."""
        consciousness_state = self.create_inputs(key, batch_size, 1, hidden_dim).squeeze(1)
        integrated_output = self.create_inputs(
            jax.random.fold_in(key, 1), batch_size, 1, hidden_dim
        ).squeeze(1)

        variables = state_manager.init(
            key,
            consciousness_state,
            integrated_output,
            deterministic=True
        )

        # Test value estimation consistency
        _, metrics1 = state_manager.apply(
            variables,
            consciousness_state,
            integrated_output,
            deterministic=True
        )
        _, metrics2 = state_manager.apply(
            variables,
            consciousness_state,
            integrated_output,
            deterministic=True
        )

        # Same input should give same value estimate
        assert jnp.allclose(metrics1['state_value'], metrics2['state_value'], rtol=1e-5)

    def test_adaptive_gating(self, state_manager, key, batch_size, hidden_dim):
        """Test adaptive gating mechanisms."""
        consciousness_state = self.create_inputs(key, batch_size, 1, hidden_dim).squeeze(1)

        # Test with different integrated outputs
        integrated_outputs = [
            self.create_inputs(jax.random.fold_in(key, i), batch_size, 1, hidden_dim).squeeze(1)
            for i in range(3)
        ]

        variables = state_manager.init(
            key,
            consciousness_state,
            integrated_outputs[0],
            deterministic=True
        )

        # Track gating behavior
        new_states = []
        for integrated_output in integrated_outputs:
            new_state, _ = state_manager.apply(
                variables,
                consciousness_state,
                integrated_output,
                deterministic=True
            )
            new_states.append(new_state)

        # Different inputs should lead to different states
        states = jnp.stack(new_states)
        for i in range(len(states)-1):
            assert not jnp.allclose(states[i], states[i+1], rtol=1e-5)
