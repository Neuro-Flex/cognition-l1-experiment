"""
Environment setup script with error handling and verification.
"""
import subprocess
import sys
import os
from typing import List, Tuple

def run_command(command: List[str]) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def setup_environment():
    """Set up the Python environment with required packages."""
    print("Setting up AI consciousness development environment...")

    # Ensure we're using Python 3.8+
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        raise RuntimeError(f"Python 3.8+ required, got {python_version.major}.{python_version.minor}")

    # Create virtual environment
    if not os.path.exists('venv'):
        print("\nCreating virtual environment...")
        returncode, stdout, stderr = run_command([sys.executable, '-m', 'venv', 'venv'])
        if returncode != 0:
            raise RuntimeError(f"Failed to create virtual environment: {stderr}")

    # Determine the pip path
    pip_path = os.path.join('venv', 'bin', 'pip') if os.name != 'nt' else os.path.join('venv', 'Scripts', 'pip')

    # Upgrade pip
    print("\nUpgrading pip...")
    returncode, stdout, stderr = run_command([pip_path, 'install', '--upgrade', 'pip'])
    if returncode != 0:
        raise RuntimeError(f"Failed to upgrade pip: {stderr}")

    # Install required packages
    packages = [
        'jax>=0.4.35',
        'jaxlib>=0.4.35',
        'flax>=0.10.2',
        'optax>=0.2.4',
        'torch',  # Latest stable version
        'transformers>=4.36.0',
        'pytest>=7.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0'
    ]

    print("\nInstalling required packages...")
    for package in packages:
        print(f"Installing {package}...")
        returncode, stdout, stderr = run_command([pip_path, 'install', '--no-cache-dir', package])
        if returncode != 0:
            raise RuntimeError(f"Failed to install {package}: {stderr}")

    # Verify installations
    print("\nVerifying installations...")
    verification_code = """
import jax
import flax
import optax
import torch
import transformers
print('JAX version:', jax.__version__)
print('Flax version:', flax.__version__)
print('Optax version:', optax.__version__)
print('PyTorch version:', torch.__version__)
print('Transformers version:', transformers.__version__)
print('Available devices:', jax.devices())
print('Environment setup completed successfully!')
"""

    python_path = os.path.join('venv', 'bin', 'python') if os.name != 'nt' else os.path.join('venv', 'Scripts', 'python')
    returncode, stdout, stderr = run_command([python_path, '-c', verification_code])

    if returncode != 0:
        raise RuntimeError(f"Environment verification failed: {stderr}")

    print("\nEnvironment setup completed successfully!")
    print(stdout)

if __name__ == '__main__':
    try:
        setup_environment()
    except Exception as e:
        print(f"Error setting up environment: {str(e)}", file=sys.stderr)
        sys.exit(1)
