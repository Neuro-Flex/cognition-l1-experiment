"""Configuration module for AI consciousness implementation.

This module handles hardware-specific configurations and model parameters.
"""
from dataclasses import dataclass
from typing import Optional, List
import jax
import torch

@dataclass
class HardwareConfig:
    """Hardware configuration settings."""
    device_type: str = "cpu"  # Current device type (cpu/gpu/tpu)
    num_devices: int = 1      # Number of available devices
    memory_limit: Optional[int] = None  # Memory limit in bytes

    @classmethod
    def from_environment(cls) -> "HardwareConfig":
        """Initialize configuration from current environment."""
        # Detect available devices
        jax_devices = jax.devices()
        torch_cuda = torch.cuda.is_available()

        return cls(
            device_type="cpu",  # Currently enforcing CPU for development
            num_devices=len(jax_devices),
            memory_limit=None  # Will be set based on system monitoring
        )

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    max_sequence_length: int = 512
    vocab_size: int = 50000
    dropout_rate: float = 0.1

    # CPU optimization parameters
    batch_size: int = 32  # Smaller batch size for CPU
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = False  # Disabled for CPU

    def optimize_for_cpu(self):
        """Adjust parameters for CPU optimization."""
        self.batch_size = min(self.batch_size, 32)
        self.mixed_precision = False
        self.gradient_accumulation_steps = max(self.gradient_accumulation_steps, 4)

@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100

    def optimize_for_cpu(self):
        """Adjust training parameters for CPU."""
        self.batch_size = 16
        self.gradient_accumulation_steps = 4
        self.eval_steps = 1000  # Less frequent evaluation on CPU
