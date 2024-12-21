"""
Main consciousness model integrating all components.
"""
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict

from .attention import GlobalWorkspace
from .memory import WorkingMemory, InformationIntegration
from .consciousness_state import CognitiveProcessIntegration, ConsciousnessStateManager

class ConsciousnessModel(nn.Module):
    """
    Complete consciousness model integrating GWT, IIT, working memory,
    and cognitive process management.
    """
    hidden_dim: int
    num_heads: int
    num_layers: int
    num_states: int
    dropout_rate: float = 0.1

    def setup(self):
        """Initialize consciousness model components."""
        # Global Workspace for conscious awareness
        self.global_workspace = GlobalWorkspace(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            dropout_rate=self.dropout_rate
        )

        # Working memory with GRU cells
        self.working_memory = WorkingMemory(
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )

        # Information integration component
        self.information_integration = InformationIntegration(
            hidden_dim=self.hidden_dim,
            num_modules=self.num_layers,
            dropout_rate=self.dropout_rate
        )

        # Cognitive process integration
        self.cognitive_integration = CognitiveProcessIntegration(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        )

        # Consciousness state management
        self.state_manager = ConsciousnessStateManager(
            hidden_dim=self.hidden_dim,
            num_states=self.num_states,
            dropout_rate=self.dropout_rate
        )

        # Fix GRUCell initialization with features parameter
        self.gru_cell = nn.GRUCell(features=self.hidden_dim)

        # Add shape alignment layer
        self.align_layer = nn.Dense(self.hidden_dim)

    def __call__(self, inputs: Dict[str, jnp.ndarray], state=None, deterministic: bool = True, consciousness_threshold: float = 0.5):
        """
        Process inputs through consciousness architecture.

        Args:
            inputs: Dictionary of input tensors for different modalities
            state: Optional previous consciousness state
            deterministic: If True, disable dropout
            consciousness_threshold: Threshold for consciousness state transitions (0.0-1.0)

        Returns:
            Updated consciousness state and metrics
        """
        # Validate and process inputs
        batch_size = next(iter(inputs.values())).shape[0]
        inputs = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in inputs.items()}

        # Initialize consciousness state if none provided
        if state is None:
            state = jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32)
        else:
            state = jnp.asarray(state, dtype=jnp.float32)

        metrics = {}

        # Global workspace processing with explicit shapes
        workspace_input = next(iter(inputs.values()))
        workspace_output, attention_weights = self.global_workspace(
            workspace_input,
            deterministic=deterministic
        )
        metrics['attention_weights'] = attention_weights

        # Working memory update with explicit shapes
        memory_output, memory_state = self.working_memory(
            workspace_output,
            initial_state=state,
            deterministic=deterministic
        )
        metrics['memory_state'] = memory_state

        # Information integration with explicit shapes
        integrated_output, phi = self.information_integration(
            memory_output,
            deterministic=deterministic
        )
        metrics['phi'] = phi

        # Cognitive process integration with consciousness threshold
        consciousness_state, attention_maps = self.cognitive_integration(
            {k: jnp.asarray(v, dtype=jnp.float32) for k, v in inputs.items()},
            deterministic=deterministic
        )
        metrics['attention_maps'] = attention_maps

        # Update consciousness state with threshold
        new_state, state_metrics = self.state_manager(
            consciousness_state,
            integrated_output,
            threshold=consciousness_threshold,
            deterministic=deterministic
        )
        metrics.update(state_metrics)

        return new_state, metrics

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'num_states': self.num_states,
            'dropout_rate': self.dropout_rate
        }

    @classmethod
    def create_default_config(cls) -> Dict[str, Any]:
        """Create default model configuration."""
        return {
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'num_states': 4,
            'dropout_rate': 0.1
        }
