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
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs, initial_state=None, mask=None, deterministic=True):
        """Process sequence through working memory."""
        batch_size = inputs.shape[0]
        if initial_state is None:
            initial_state = jnp.zeros((batch_size, self.hidden_dim))

        # Define RNN cell outside scan_fn
        rnn_cell = nn.LSTMCell(features=self.hidden_dim)

        # Initialize both hidden and cell states
        if initial_state is None:
            initial_h = jnp.zeros((batch_size, self.hidden_dim))
            initial_c = jnp.zeros((batch_size, self.hidden_dim))
        else:
            initial_h = initial_state
            initial_c = jnp.zeros_like(initial_state)

        # Generate a PRNG key for LSTM initialization
        key = self.make_rng('params')

        # Process sequence using pure function for JAX compatibility
        def scan_fn(carry: Tuple, x):
            lstm_state, prev_h = carry
            new_lstm_state, new_h = rnn_cell(lstm_state, x)
            return (new_lstm_state, new_h), new_h

        # Initialize proper LSTM state
        input_shape = (batch_size, self.hidden_dim)
        init_lstm_carry = (
            nn.LSTMCell.initialize_carry(key, (self.hidden_dim,), self.hidden_dim),
            initial_h
        )

        # Apply dropout if not in deterministic mode
        if not deterministic:
            dropout_key = self.make_rng('dropout')
            inputs = nn.Dropout(rate=self.dropout_rate)(inputs, deterministic=False, rng=dropout_key)

        # Scan over the sequence
        (final_lstm_state, final_h), outputs = jax.lax.scan(scan_fn, init_lstm_carry, inputs)

        return outputs, final_lstm_state

class InformationIntegration(nn.Module):
    """
    Implementation of Information Integration Theory (IIT) components.
    """
    hidden_dim: int
    num_modules: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        # Project all inputs to same dimension if needed
        if inputs.shape[-1] != self.hidden_dim:
            inputs = nn.Dense(features=self.hidden_dim)(inputs)
            
        # Apply layer normalization
        x = nn.LayerNorm()(inputs)
        
        # Apply self-attention across modules
        y = nn.MultiHeadAttention(
            num_heads=4,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x, x, x)
        
        if not deterministic:
            y = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(y)
            
        # Add residual connection
        output = x + y
        
        # Calculate integration metric (phi)
        # Shape: (batch_size,)
        phi = jnp.mean(jnp.abs(output), axis=(-2, -1))
        
        return output, phi
