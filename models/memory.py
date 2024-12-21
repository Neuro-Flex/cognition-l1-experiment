import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple

class GRUCell(nn.Module):
    """
    Gated Recurrent Unit cell with consciousness-aware gating mechanisms.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, x, h):
        """
        Apply GRU cell with consciousness-aware updates.

        Args:
            x: Input tensor [batch_size, input_dim]
            h: Hidden state [batch_size, hidden_dim]
        """
        # Ensure inputs are float32 arrays
        x = jnp.asarray(x, dtype=jnp.float32)
        h = jnp.asarray(h, dtype=jnp.float32)

        # Concatenate inputs for efficiency
        inputs = jnp.concatenate([x, h], axis=-1)

        # Create update gate
        z = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            name='update'
        )(inputs)
        z = nn.sigmoid(z)

        # Create reset gate
        r = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            name='reset'
        )(inputs)
        r = nn.sigmoid(r)

        # Create candidate activation
        h_reset = r * h
        h_concat = jnp.concatenate([x, h_reset], axis=-1)
        h_tilde = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros,
            name='candidate'
        )(h_concat)
        h_tilde = jnp.tanh(h_tilde)

        # Final update
        h_new = jnp.clip((1.0 - z) * h + z * h_tilde, -1.0, 1.0)

        return h_new

class WorkingMemory(nn.Module):
    """
    Working memory component with context-aware gating for consciousness.
    """
    hidden_dim: int
    dropout_rate: float = 0.1

    def setup(self):
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.gru = GRUCell(hidden_dim=self.hidden_dim)

    @nn.compact
    def __call__(self, inputs, initial_state=None, mask=None, deterministic=True):
        """
        Process sequence through working memory.

        Args:
            inputs: Input sequence [batch, seq_len, input_dim]
            initial_state: Optional initial hidden state
            mask: Boolean mask [batch, seq_len]
            deterministic: If True, disable dropout
        """
        batch_size = inputs.shape[0]
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.hidden_dim))

        # Define RNN cell outside scan_fn
        rnn_cell = nn.LSTMCell(features=self.hidden_dim)

        # Process sequence using pure function for JAX compatibility
        def scan_fn(h, x):
            h_new, y = rnn_cell(h, x)
            return (h_new, y), y

        # Ensure inputs and state are float32
        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        initial_state = jnp.asarray(initial_state, dtype=jnp.float32)

        # Use scan with explicit axis for sequence processing
        (final_state, _), outputs = jax.lax.scan(
            scan_fn,
            init=(initial_state, None),
            xs=inputs.swapaxes(0, 1)
        )
        outputs = outputs.swapaxes(0, 1)

        # Apply dropout using Flax's deterministic dropout
        if not deterministic:
            outputs = self.dropout(outputs, deterministic=deterministic)

        return outputs, final_state

class InformationIntegration(nn.Module):
    """
    Implementation of Information Integration Theory (IIT) components.
    """
    hidden_dim: int
    num_modules: int
    dropout_rate: float = 0.1

    def setup(self):
        self.layer_norm = nn.LayerNorm()
        self.dense1 = nn.Dense(self.hidden_dim)
        self.dense2 = nn.Dense(self.hidden_dim)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, inputs, deterministic=True):
        """
        Integrate information across modules using IIT principles.

        Args:
            inputs: Input tensor [batch, num_modules, input_dim]
            deterministic: If True, disable dropout
        """
        # Layer normalization for stability
        x = self.layer_norm(inputs)

        # Dense network with GELU activation and residual connection
        y = self.dense1(x)
        y = nn.gelu(y)

        # Apply dropout
        if not deterministic:
            y = self.dropout(y)

        y = self.dense2(y)
        if not deterministic:
            y = self.dropout(y)

        # Residual connection
        output = y + x

        # Compute information integration metric (Φ)
        def compute_entropy(p):
            # Ensure proper broadcasting for entropy calculation
            eps = 1e-12
            p = jnp.clip(p, eps, 1.0 - eps)
            return -jnp.sum(p * jnp.log(p), axis=-1, keepdims=True)

        # Reshape tensors to ensure compatible broadcasting
        module_probs = nn.softmax(output, axis=-1)
        module_entropies = jax.vmap(compute_entropy)(module_probs)
        avg_module_entropy = jnp.mean(module_entropies, axis=1)  # [batch_size, 1]

        system_logits = jnp.mean(output, axis=1)
        system_probs = nn.softmax(system_logits, axis=-1)
        system_entropy = compute_entropy(system_probs)  # [batch_size, 1]

        # Ensure compatible shapes for subtraction
        phi = jnp.squeeze(avg_module_entropy - system_entropy, axis=-1)

        return output, phi
