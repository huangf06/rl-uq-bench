"""
UQ Pipeline: Uncertainty Quantification Evaluation System

A modular, scalable pipeline for evaluating uncertainty quantification methods
in reinforcement learning environments.
"""

__version__ = "1.0.0"
__author__ = "UQ Pipeline Team"
__description__ = "Uncertainty Quantification Evaluation Pipeline for Reinforcement Learning"

from .utils.context import ExperimentContext
from .runner import main as run_pipeline

__all__ = [
    "ExperimentContext",
    "run_pipeline"
]