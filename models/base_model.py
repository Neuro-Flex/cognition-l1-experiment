import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from core.config import ModelConfig, TrainingConfig
from core.hardware import HardwareManager

class BaseAttention(nn.Module):
    """Multi-head attention mechanism optimized for CPU."""
    hidden_size: int
    num_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.query = nn.Dense(self.hidden_size)
        self.key = nn.Dense(self.hidden_size)
        self.value = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.hidden_size)

    def __call__(self, x, mask=None, deterministic=True):
        batch_size = x.shape[0]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention
        head_dim = self.hidden_size // self.num_heads
        q = q.reshape(batch_size, -1, self.num_heads, head_dim)
        k = k.reshape(batch_size, -1, self.num_heads, head_dim)
        v = v.reshape(batch_size, -1, self.num_heads, head_dim)

        # Compute attention scores
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k)
        scores = scores / jnp.sqrt(head_dim)

        if mask is not None:
            # Ensure mask is broadcastable
            mask = jnp.expand_dims(mask, axis=(-3, -2))
            scores = scores + mask * -1e9

        weights = jax.nn.softmax(scores, axis=-1)

        if not deterministic:
            weights = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(weights)

        # Compute weighted sum
        output = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        output = output.reshape(batch_size, -1, self.hidden_size)

        return self.output(output)

class BaseModel(nn.Module):
    """Base transformer model with CPU optimizations."""
    config: ModelConfig

    def setup(self):
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size
        )

        self.encoder_layers = [
            BaseAttention(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                dropout_rate=self.config.dropout_rate
            )
            for _ in range(self.config.num_hidden_layers)
        ]

        self.pooler = nn.Dense(self.config.hidden_size)

    def __call__(self, input_ids, attention_mask=None, deterministic=True):
        assert input_ids.ndim == 2, "input_ids must be of shape (batch_size, seq_len)"
        assert input_ids.dtype == jnp.int32, "input_ids must be integers"

        x = self.embeddings(input_ids)

        if attention_mask is not None:
            assert attention_mask.shape == input_ids.shape, "attention_mask shape must match input_ids"

        for layer in self.encoder_layers:
            x = layer(x, mask=attention_mask, deterministic=deterministic)

        pooled = self.pooler(x[:, 0])
        return x, pooled

def create_train_state(
    model: BaseModel,
    config: TrainingConfig,
    rng: jax.random.PRNGKey
) -> train_state.TrainState:
    """Creates initial training state with CPU-optimized settings."""
    params = model.init(rng, jnp.ones((1, config.max_sequence_length), dtype=jnp.int32))

    # CPU-optimized learning rate schedule
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps
    )

    # Optimizer with CPU-specific settings
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=schedule_fn,
            weight_decay=config.weight_decay,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )