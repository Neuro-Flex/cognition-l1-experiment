"""Initialization module for AI consciousness implementation."""
import os
import jax
import jax.numpy as jnp
from typing import Tuple
import logging

from core.config import ModelConfig, TrainingConfig, HardwareConfig
from core.hardware import HardwareManager
from models.base_model import BaseModel, create_train_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_system() -> Tuple[BaseModel, HardwareConfig, ModelConfig, TrainingConfig]:
    """Initialize the AI consciousness system with CPU optimization."""
    # Initialize hardware manager
    hw_manager = HardwareManager()
    hw_manager.setup_environment()

    # Get hardware configuration
    hw_config = HardwareConfig.from_environment()

    # Initialize model configuration with CPU optimizations
    model_config = ModelConfig()
    model_config.optimize_for_cpu()

    # Initialize training configuration
    train_config = TrainingConfig()
    train_config.optimize_for_cpu()

    # Create base model
    model = BaseModel(config=model_config)

    # Initialize random seed
    rng = jax.random.PRNGKey(0)

    # Test basic model functionality
    batch_size = model_config.batch_size
    seq_length = model_config.max_sequence_length
    test_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)

    try:
        # Test forward pass
        params = model.init(rng, test_input)
        output, pooled = model.apply(params, test_input)
        logger.info("Model initialization successful")
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Pooled output shape: {pooled.shape}")
        logger.info(f"Params: {params}")

        # Log hardware configuration
        hw_info = hw_manager.get_hardware_info()
        logger.info("\nHardware Configuration:")
        for key, value in hw_info.items():
            logger.info(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

    return model, hw_config, model_config, train_config

if __name__ == "__main__":
    initialize_system()
