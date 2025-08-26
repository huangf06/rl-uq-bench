# RL-UQ-Bench: Uncertainty Quantification Benchmark for Deep RL

This is a comprehensive benchmarking framework for evaluating uncertainty quantification (UQ) methods in deep reinforcement learning agents.

## Project Overview

RL-UQ-Bench provides systematic evaluation of multiple UQ methods (Bootstrapped DQN, QR-DQN, MC Dropout, hybrid approaches) across various noise settings and environments using rigorous statistical evaluation.

## Key Components

- **Training Framework**: Extended RL Baselines3 Zoo with UQ capabilities
- **Evaluation Pipeline**: Multi-stage UQ evaluation with comprehensive metrics  
- **Experiment Configurations**: Standardized setups for reproducible benchmarks
- **Statistical Analysis**: Rigorous evaluation with multiple seeds and significance testing

## Evaluation Metrics

Primary metrics for UQ quality assessment:
- **ECE/ACE**: Expected/Adaptive Calibration Error
- **CRPS**: Continuous Ranked Probability Score  
- **Coverage**: Prediction interval coverage
- **WIS**: Weighted Interval Score

## Usage

This framework supports both individual method evaluation and comprehensive benchmarking across multiple methods, environments, and noise conditions.

For detailed usage instructions, see the main README.md file.