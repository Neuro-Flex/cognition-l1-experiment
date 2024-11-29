"""Test script for verifying environment setup."""
import jax
import jax.numpy as jnp
from core.initialize import initialize_system
from core.hardware import HardwareManager

print('Testing environment setup...\n')

# Test hardware manager
hw_manager = HardwareManager()
hw_info = hw_manager.get_hardware_info()
print('Hardware Configuration:')
for key, value in hw_info.items():
    print(f'{key}: {value}')

print('\nInitializing system...')
model, hw_config, model_config, train_config = initialize_system()
print('\nEnvironment setup completed successfully')
