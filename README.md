# RL-UQ-Bench: Uncertainty Quantification Benchmark for Deep Reinforcement Learning

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

A comprehensive evaluation framework for uncertainty quantification (UQ) methods in deep reinforcement learning agents.

## 🎯 Overview

RL-UQ-Bench provides a systematic benchmarking framework to evaluate multiple uncertainty quantification methods in deep RL agents across various noise settings and environments. The framework supports rigorous statistical evaluation using metrics like Expected Calibration Error (ECE), Continuous Ranked Probability Score (CRPS), Coverage, and Weighted Interval Score (WIS).

### Key Features

- **Multiple UQ Methods**: Bootstrapped DQN, QR-DQN, MC Dropout, and hybrid approaches
- **Comprehensive Evaluation**: ECE, CRPS, Coverage, WIS metrics with statistical significance testing  
- **Noise Robustness Testing**: Systematic evaluation across different noise levels
- **Reproducible Experiments**: Standardized configurations with multiple random seeds
- **Extensible Framework**: Easy to add new UQ methods and environments

### Supported UQ Methods

- **Bootstrapped DQN**: Ensemble-based uncertainty estimation
- **QR-DQN (Quantile Regression DQN)**: Distributional RL approach  
- **MC Dropout**: Variational inference with dropout
- **QR-Bootstrap Hybrid**: Combined quantile regression and bootstrapping
- **Standard DQN**: Baseline without UQ

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/huangf06/rl-uq-bench.git
cd rl-uq-bench
pip install -e .
```

### Basic Usage

#### Training a UQ-enabled agent:

```bash
python train.py --algo qrdqn --env LunarLander-v2 --noise-std 0.1 --seed 42
```

#### Running UQ evaluation:

```bash
python -m uq_pipeline.runner --config uq_pipeline/configs/complete_multi_method_experiment.yml
```

#### Generating benchmark results:

```bash
# Run complete benchmark across all methods and noise levels
python experiments/uncertainty_degradation/submit_all_experiments.sh
```

## 📊 Benchmark Results

The framework evaluates UQ methods across multiple dimensions:

- **Calibration Quality**: How well predicted uncertainties match actual errors
- **Noise Robustness**: Performance degradation under different noise levels  
- **Computational Efficiency**: Training time and inference overhead
- **Statistical Significance**: Rigorous statistical testing across seeds

Key findings from our benchmark:
- QR-DQN shows superior calibration in clean environments
- Bootstrapped DQN maintains robustness under high noise
- MC Dropout provides computational efficiency but lower calibration quality
- Hybrid methods can combine benefits of multiple approaches

## 🔧 Configuration

### Experiment Configuration

Experiments are configured via YAML files in `experiments/` and `uq_pipeline/configs/`:

```yaml
# Example configuration
experiment:
  name: "noise_robustness_test"
  methods: ["dqn", "qrdqn", "bootstrapped_dqn", "mcdropout_dqn"]
  environments: ["LunarLander-v2"]
  noise_levels: [0.0, 0.05, 0.1, 0.2]
  seeds: [0, 1, 2, 3, 4]
  
evaluation:
  metrics: ["ece", "crps", "coverage", "wis"]
  statistical_tests: true
```

### Hyperparameters

Optimized hyperparameters for each method are provided in `hyperparams/`:

- `hyperparams/qrdqn.yml`: QR-DQN parameters
- `hyperparams/bootstrapped_dqn.yml`: Bootstrap ensemble settings
- `hyperparams/mcdropout_dqn.yml`: MC Dropout configurations

## 📁 Project Structure

```
rl-uq-bench/
├── rl_zoo3/                 # Core RL training framework
├── uq_pipeline/            # UQ evaluation pipeline  
├── experiments/            # Experiment configurations
├── hyperparams/           # Optimized hyperparameters
├── wrappers/              # Environment wrappers (noise injection)
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🧪 Evaluation Pipeline

The UQ evaluation pipeline consists of multiple stages:

1. **Stage 0**: Configuration validation
2. **Stage 1**: Dataset building from trained models  
3. **Stage 2**: Performance evaluation
4. **Stage 3**: Q-value extraction
5. **Stage 4**: UQ metrics calculation
6. **Stage 5**: Calibration analysis
7. **Stage 6**: Statistical analysis and reporting

## 🎯 Supported Environments

Currently tested environments:
- **LunarLander-v2**: Classic control with discrete actions
- **CartPole-v1**: Simple control benchmark
- **MountainCar-v0**: Sparse reward environment

Easy to extend to other Gymnasium/OpenAI Gym environments.

## 📈 Metrics

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Reliability of uncertainty estimates
- **Adaptive Calibration Error (ACE)**: Adaptive binning approach

### Distributional Metrics  
- **Continuous Ranked Probability Score (CRPS)**: Proper scoring rule
- **Coverage**: Percentage of true values within prediction intervals
- **Weighted Interval Score (WIS)**: Interval prediction quality

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

Areas for contribution:
- New UQ methods
- Additional environments
- Improved evaluation metrics
- Performance optimizations

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{rl_uq_bench_2024,
  title={RL-UQ-Bench: Uncertainty Quantification Benchmark for Deep Reinforcement Learning},
  author={Huang, F.},
  year={2024},
  url={https://github.com/huangf06/rl-uq-bench}
}
```

## 🏗️ Built On

This framework extends [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) with uncertainty quantification capabilities:

- **Stable Baselines3**: Core RL algorithms
- **SB3-Contrib**: Additional RL algorithms  
- **Gymnasium**: Environment interface
- **PyTorch**: Deep learning backend

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the excellent [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- Inspired by uncertainty quantification research in deep learning
- Thanks to the open source RL community