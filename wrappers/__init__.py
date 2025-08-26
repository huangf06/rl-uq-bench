"""
Custom Gymnasium environment wrappers.

Provides wrappers for enhancing environment complexity and testing agent robustness.
"""

from .noise import GaussianObsNoise

__all__ = ["GaussianObsNoise"]

__version__ = "1.1.0" 