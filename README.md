# RL-UQ-Bench: Uncertainty Quantification Benchmark for Deep Reinforcement Learning

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

A comprehensive benchmarking framework for uncertainty quantification (UQ) methods in deep reinforcement learning, providing systematic evaluation across environmental perturbations and standardized metrics.

## 🎯 Overview

RL-UQ-Bench addresses critical gaps in uncertainty quantification research for deep RL by providing a standardized evaluation framework. Based on rigorous academic research, this benchmark enables fair comparison of UQ methods under controlled experimental conditions, with particular attention to real-world environmental factors like observation noise.

### Key Research Findings

This systematic benchmark of 150 independent training runs reveals:

- **QR-DQN superiority**: Distributional learning (QR-DQN) consistently outperforms ensemble and dropout methods across all metrics
- **Noise paradox discovery**: Bootstrapped DQN exhibits counterintuitive improvement under moderate noise (σ ≈ 0.025)
- **Method-dependent calibration**: Post-hoc calibration benefits ensemble/dropout methods but shows mixed effects for QR-DQN
- **Standardized evaluation**: Reproducible framework enabling reliable cross-method comparisons

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
# QR-DQN with noise
python train.py --algo qrdqn --env LunarLander-v3 --noise-std 0.025 --seed 42

# Bootstrapped DQN ensemble  
python train.py --algo bootstrapped_dqn --env LunarLander-v3 --seed 42

# MC Dropout DQN
python train.py --algo mcdropout_dqn --env LunarLander-v3 --noise-std 0.1 --seed 42
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

Systematic evaluation (3 methods × 5 noise levels × 10 seeds = 150 training runs) reveals:

### Aggregate UQ Quality
| Method | CRPS | WIS | ACE | Coverage (90%) |
|--------|------|-----|-----|----------------|
| **QR-DQN** | **17.7** ± 8.7 | **150.2** ± 89.3 | **0.14** ± 0.12 | **83** ± 16 |
| Bootstrapped DQN | 23.6 ± 12.6 | 365.0 ± 227.8 | 0.50 ± 0.17 | 31 ± 22 |
| MC Dropout DQN | 26.5 ± 10.9 | 392.4 ± 171.2 | 0.49 ± 0.12 | 33 ± 14 |

**Key Findings:**
- **QR-DQN dominance**: 31% lower CRPS than Bootstrapped DQN, 45% lower than MC Dropout
- **Statistical significance**: All differences p < 0.001 with medium-large effect sizes
- **Coverage quality**: QR-DQN achieves 83% coverage closest to nominal 90% target

### Noise Robustness Analysis
- **QR-DQN**: Robust performance across all noise levels (σ = 0.000 to 0.100)
- **Bootstrapped DQN**: Exhibits "noise paradox" - optimal performance at σ ≈ 0.025
- **MC Dropout**: Monotonic degradation with increasing noise

## 🔧 Configuration

### Experiment Configuration

Experiments are configured via YAML files in `experiments/` and `uq_pipeline/configs/`:

```yaml
# Example configuration
experiment:
  name: "noise_robustness_test"
  methods: ["dqn", "qrdqn", "bootstrapped_dqn", "mcdropout_dqn"]
  environments: ["LunarLander-v3"]
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

The standardized 6-stage evaluation pipeline ensures reproducible and comprehensive UQ assessment:

1. **Model Training**: Standardized architectures and hyperparameters
2. **Task Performance**: Verification of meaningful policy learning
3. **Data Generation**: Controlled evaluation episodes with logged uncertainties
4. **Uncertainty Representation**: Method-specific uncertainty extraction
5. **Calibration**: Post-hoc calibration with bias/variance adjustment
6. **Metric Computation**: CRPS, ACE, WIS, Coverage evaluation

### Experimental Configuration
- **Environment**: LunarLander-v3 (continuous 8-dimensional state space, 4 discrete actions)
- **Design**: Full factorial 3×5×10 (methods × noise levels × seeds)
- **Training**: Standardized hyperparameters and network architectures
- **Evaluation**: 100-episode evaluation windows per trained agent

## 🎯 Supported Environments

Currently tested environments:
- **LunarLander-v3**: Classic control with discrete actions
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
@software{rl_uq_bench,
  author = {Huang, Fei},
  title = {RL-UQ-Bench: Uncertainty Quantification Benchmark for Deep Reinforcement Learning},
  url = {https://github.com/huangf06/rl-uq-bench},
  year = {2025}
}

@mastersthesis{huang2025uq_deep_rl,
  title = {Uncertainty Quantification in Deep Reinforcement Learning: Systematic Benchmarks, Noise Paradox, and Calibration Complexity},
  author = {Huang, Fei},
  year = {2025},
  school = {Vrije Universiteit Amsterdam},
  type = {Master's Thesis}
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