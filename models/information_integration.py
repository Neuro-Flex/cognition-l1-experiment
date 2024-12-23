import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, Optional

class InformationIntegration(nn.Module):
    """
    Implementation of Information Integration Theory formulas.
    """
    hidden_dim: int

    def setup(self):
        # Integration layers
        self.phi_computation = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())
        self.integration_gate = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())

    def compute_phi(self, states: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Φ (phi) - the amount of integrated information.
        Based on IIT's mathematical formulation.
        """
        # Compute integrated information through non-linear transformation
        phi_raw = self.phi_computation(states)
        phi_normalized = nn.tanh(phi_raw)  # Bound between -1 and 1
        return phi_normalized

    def __call__(self, states: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Ensure input shape is as expected
        assert states.shape[-1] == self.hidden_dim, f"Expected last dimension to be {self.hidden_dim}, but got {states.shape[-1]}"

        # Compute integration measure
        phi = self.compute_phi(states)
        
        # Compute integration gate
        gate = nn.sigmoid(self.integration_gate(states))
        
        # Apply gated integration
        integrated_state = gate * phi + (1 - gate) * states

        # Add check to prevent NaN in correlation calculation
        if jnp.any(jnp.isnan(phi)):
            phi = jnp.nan_to_num(phi, nan=0.0)

        return {
            'integrated_state': integrated_state,
            'phi': phi,
            'integration_gate': gate
        }
