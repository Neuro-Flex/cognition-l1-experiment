"""
Setup configuration for AI consciousness implementation.
"""

from setuptools import setup, find_packages

setup(
    name="ai_consciousness",
    version="0.1.0",
    description="AI consciousness implementation using JAX, Flax, and Optax",
    author="Devin",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "jax[cuda]>=0.4.35",
        "jaxlib>=0.4.35",
        "flax>=0.10.2",
        "optax>=0.2.4",
        "torch>=2.5.1",
        "transformers>=4.37.2",
        "datasets>=2.16.1",
        "sympy>=1.12",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.2.0",
        "pytest>=7.4.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
