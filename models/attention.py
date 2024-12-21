import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple

class ConsciousnessAttention(nn.Module):
    """
    Multi-head attention mechanism for consciousness modeling based on Global Workspace Theory.
    Implements scaled dot-product attention with consciousness-aware broadcasting.
    """
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, mask=None, deterministic=True):
        """
        Apply consciousness-aware attention mechanism.

        Args:
            inputs_q: Query input of shape [batch, length, depth]
            inputs_kv: Key/value input of shape [batch, length, depth]
            mask: Boolean mask of shape [batch, length]
            deterministic: If True, disable dropout

        Returns:
            Output of shape [batch, length, depth]
        """
        depth = self.num_heads * self.head_dim

        # Project inputs to queries, keys, and values
        query = nn.Dense(depth, name='query')(inputs_q)
        key = nn.Dense(depth, name='key')(inputs_kv)
        value = nn.Dense(depth, name='value')(inputs_kv)

        # Reshape for multi-head attention
        batch_size = query.shape[0]
        query = query.reshape(batch_size, -1, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, -1, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        depth_scaling = jnp.sqrt(self.head_dim).astype(key.dtype)
        attention_logits = jnp.einsum('bqhd,bkhd->bhqk', query, key)
        attention_logits = attention_logits / depth_scaling

        if mask is not None:
            # Expand mask to match the attention logits shape
            attention_mask = mask[:, None, None, :]
            attention_logits = jnp.where(attention_mask, attention_logits, -1e10)

        attention_weights = nn.softmax(attention_logits, axis=-1)

        # Apply dropout to attention weights if not deterministic
        if not deterministic:
            attention_weights = nn.Dropout(
                rate=self.attention_dropout_rate,
            )(attention_weights, deterministic=deterministic)

        # Compute attention output
        attention_output = jnp.einsum('bhqk,bkhd->bqhd', attention_weights, value)

        # Reshape back to original dimensions
        attention_output = attention_output.reshape(batch_size, -1, depth)

        # Final projection
        output = nn.Dense(inputs_q.shape[-1], name='output')(attention_output)

        # Apply output dropout if not deterministic
        if not deterministic:
            output = nn.Dropout(rate=self.dropout_rate)(
                output, deterministic=deterministic
            )

        # Adjust output shape
        output = output.reshape(inputs_q.shape)

        # Residual connection
        output = inputs_q + output

        return output, attention_weights

class GlobalWorkspace(nn.Module):
    """
    Implementation of Global Workspace Theory for consciousness modeling.
    Integrates information from multiple cognitive processes through attention.
    """
    hidden_dim: int
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=True):
        """
        Process inputs through global workspace architecture.

        Args:
            inputs: Input tensor of shape [batch, length, depth]
            mask: Boolean mask of shape [batch, length]
            deterministic: If True, disable dropout

        Returns:
            Processed output maintaining conscious awareness
        """
        # Layer normalization for stability
        x = nn.LayerNorm()(inputs)

        # Self-attention for global workspace broadcasting
        attention = ConsciousnessAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )

        # Apply attention with residual connection
        attended_output, attention_weights = attention(x, x, mask, deterministic)
        x = x + attended_output  # Residual connection

        # Feed-forward network for information integration
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.hidden_dim, name='ff1')(y)
        y = nn.gelu(y)
        y = nn.Dense(inputs.shape[-1], name='ff2')(y)

        # Apply dropout to feed-forward output if not deterministic
        if not deterministic:
            y = nn.Dropout(rate=self.dropout_rate)(
                y, deterministic=deterministic
            )

        # Residual connection for maintaining information flow
        output = x + y

        return output, attention_weights