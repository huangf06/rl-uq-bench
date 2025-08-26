"""
Configuration Validator for UQ Pipeline
Provides comprehensive validation for configuration files
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging


class ConfigValidator:
    """Comprehensive configuration validator"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.errors = []
        self.warnings = []
    
    def validate_full_config(self, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Perform comprehensive validation of configuration
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Core validation
        self._validate_experiment_info(config)
        self._validate_environment_config(config)
        self._validate_algorithm_config(config)
        self._validate_data_config(config)
        self._validate_metrics_config(config)
        self._validate_paths_config(config)
        self._validate_execution_config(config)
        
        # Cross-validation
        self._validate_algorithm_model_availability(config)
        self._validate_metric_algorithm_compatibility(config)
        self._estimate_resource_requirements(config)
        
        return self.errors, self.warnings
    
    def _validate_experiment_info(self, config: Dict[str, Any]):
        """Validate experiment information"""
        experiment = config.get('experiment', {})
        
        if not experiment.get('name'):
            self.errors.append("experiment.name is required")
        elif not isinstance(experiment['name'], str):
            self.errors.append("experiment.name must be a string")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', experiment['name']):
            self.errors.append("experiment.name can only contain alphanumeric characters, underscores, and hyphens")
        
        if experiment.get('version') and not isinstance(experiment['version'], str):
            self.errors.append("experiment.version must be a string")
    
    def _validate_environment_config(self, config: Dict[str, Any]):
        """Validate environment configuration"""
        env_config = config.get('environment', {})
        
        # Environment ID validation
        env_id = env_config.get('id')
        if not env_id:
            self.errors.append("environment.id is required")
        elif not isinstance(env_id, str):
            self.errors.append("environment.id must be a string")
        
        # Success threshold validation
        threshold = env_config.get('success_threshold')
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                self.errors.append("environment.success_threshold must be a number")
            elif threshold <= 0:
                self.errors.append("environment.success_threshold must be positive")
        
        # Noise levels validation
        noise_levels = env_config.get('noise_levels', {})
        if noise_levels and not isinstance(noise_levels, dict):
            self.errors.append("environment.noise_levels must be a dictionary")
        else:
            for name, value in noise_levels.items():
                if not isinstance(value, (int, float)):
                    self.errors.append(f"environment.noise_levels.{name} must be a number")
                elif value < 0:
                    self.errors.append(f"environment.noise_levels.{name} must be non-negative")
        
        # Active noise levels validation
        active_levels = env_config.get('active_noise_levels', [])
        if active_levels:
            if not isinstance(active_levels, list):
                self.errors.append("environment.active_noise_levels must be a list")
            else:
                for level in active_levels:
                    if level not in noise_levels:
                        self.errors.append(f"environment.active_noise_levels contains unknown level: {level}")
    
    def _validate_algorithm_config(self, config: Dict[str, Any]):
        """Validate algorithm configuration"""
        algo_config = config.get('algorithms', {})
        
        # Active algorithms validation
        active_algos = algo_config.get('active', [])
        if not active_algos:
            self.errors.append("algorithms.active is required and must be non-empty")
        elif not isinstance(active_algos, list):
            self.errors.append("algorithms.active must be a list")
        else:
            valid_algorithms = ['dqn', 'qrdqn', 'bootstrapped_dqn', 'mcdropout_dqn']
            for algo in active_algos:
                if algo not in valid_algorithms:
                    self.errors.append(f"Unknown algorithm: {algo}. Valid options: {valid_algorithms}")
        
        # Algorithm-specific config validation
        algo_configs = algo_config.get('configs', {})
        if not isinstance(algo_configs, dict):
            self.errors.append("algorithms.configs must be a dictionary")
        else:
            for algo in active_algos:
                if algo in algo_configs:
                    self._validate_single_algorithm_config(algo, algo_configs[algo])
                else:
                    self.warnings.append(f"No configuration found for active algorithm: {algo}")
    
    def _validate_single_algorithm_config(self, algorithm: str, config: Dict[str, Any]):
        """Validate configuration for a single algorithm"""
        if algorithm == 'qrdqn':
            n_quantiles = config.get('n_quantiles')
            if n_quantiles is not None:
                if not isinstance(n_quantiles, int):
                    self.errors.append(f"algorithms.configs.{algorithm}.n_quantiles must be an integer")
                elif n_quantiles < 3 or n_quantiles > 1000:
                    self.warnings.append(f"algorithms.configs.{algorithm}.n_quantiles ({n_quantiles}) is outside typical range [3, 1000]")
        
        elif algorithm == 'bootstrapped_dqn':
            ensemble_size = config.get('ensemble_size')
            if ensemble_size is not None:
                if not isinstance(ensemble_size, int):
                    self.errors.append(f"algorithms.configs.{algorithm}.ensemble_size must be an integer")
                elif ensemble_size < 2 or ensemble_size > 50:
                    self.warnings.append(f"algorithms.configs.{algorithm}.ensemble_size ({ensemble_size}) is outside typical range [2, 50]")
        
        elif algorithm == 'mcdropout_dqn':
            dropout_rate = config.get('dropout_rate')
            if dropout_rate is not None:
                if not isinstance(dropout_rate, (int, float)):
                    self.errors.append(f"algorithms.configs.{algorithm}.dropout_rate must be a number")
                elif dropout_rate <= 0 or dropout_rate >= 1:
                    self.errors.append(f"algorithms.configs.{algorithm}.dropout_rate must be between 0 and 1")
            
            mc_samples = config.get('mc_samples')
            if mc_samples is not None:
                if not isinstance(mc_samples, int):
                    self.errors.append(f"algorithms.configs.{algorithm}.mc_samples must be an integer")
                elif mc_samples < 1 or mc_samples > 1000:
                    self.warnings.append(f"algorithms.configs.{algorithm}.mc_samples ({mc_samples}) is outside typical range [1, 1000]")
    
    def _validate_data_config(self, config: Dict[str, Any]):
        """Validate data configuration"""
        data_config = config.get('data', {})
        
        # Seeds validation
        seeds = data_config.get('seeds', [])
        if not seeds:
            self.errors.append("data.seeds is required and must be non-empty")
        elif not isinstance(seeds, list):
            self.errors.append("data.seeds must be a list")
        else:
            for seed in seeds:
                if not isinstance(seed, int):
                    self.errors.append("All seeds in data.seeds must be integers")
                elif seed < 0:
                    self.errors.append("All seeds must be non-negative")
        
        # Episode count validation
        n_episodes = data_config.get('n_episodes_per_seed')
        if n_episodes is not None:
            if not isinstance(n_episodes, int):
                self.errors.append("data.n_episodes_per_seed must be an integer")
            elif n_episodes <= 0:
                self.errors.append("data.n_episodes_per_seed must be positive")
            elif n_episodes > 1000:
                self.warnings.append(f"data.n_episodes_per_seed ({n_episodes}) is quite large, may cause memory issues")
        
        # MC samples validation
        mc_samples = data_config.get('n_mc_samples')
        if mc_samples is not None:
            if not isinstance(mc_samples, int):
                self.errors.append("data.n_mc_samples must be an integer")
            elif mc_samples <= 0:
                self.errors.append("data.n_mc_samples must be positive")
    
    def _validate_metrics_config(self, config: Dict[str, Any]):
        """Validate metrics configuration"""
        metrics = config.get('metrics', {})
        if not metrics:
            self.warnings.append("No metrics configuration found, using defaults")
            return
        
        if not isinstance(metrics, dict):
            self.errors.append("metrics must be a dictionary")
            return
        
        # Validate metric categories
        valid_categories = ['distributional', 'interval_based', 'calibration', 'performance']
        for category in metrics.keys():
            if category not in valid_categories:
                self.warnings.append(f"Unknown metric category: {category}")
        
        # Validate specific metrics
        if 'calibration' in metrics:
            calib_metrics = metrics['calibration']
            if 'ece' in calib_metrics:
                ece_config = calib_metrics['ece']
                if isinstance(ece_config, dict):
                    n_bins = ece_config.get('n_bins')
                    if n_bins and (not isinstance(n_bins, int) or n_bins <= 0):
                        self.errors.append("metrics.calibration.ece.n_bins must be a positive integer")
    
    def _validate_paths_config(self, config: Dict[str, Any]):
        """Validate paths configuration"""
        paths = config.get('paths', {})
        if not paths:
            self.warnings.append("No paths configuration found")
            return
        
        required_paths = ['data_dir', 'results_dir']
        for path_name in required_paths:
            path_value = paths.get(path_name)
            if not path_value:
                self.errors.append(f"paths.{path_name} is required")
            elif not isinstance(path_value, (str, Path)):
                self.errors.append(f"paths.{path_name} must be a string or Path")
        
        # Check if models directory exists
        models_dir = paths.get('models_dir')
        if models_dir and not Path(models_dir).exists():
            self.warnings.append(f"Models directory does not exist: {models_dir}")
    
    def _validate_execution_config(self, config: Dict[str, Any]):
        """Validate execution configuration"""
        execution = config.get('execution', {})
        if not execution:
            return
        
        # Parallel execution validation
        parallel_config = execution.get('parallel', {})
        if parallel_config.get('enabled'):
            max_workers = parallel_config.get('max_workers')
            if max_workers and (not isinstance(max_workers, int) or max_workers <= 0):
                self.errors.append("execution.parallel.max_workers must be a positive integer")
        
        # Error handling validation
        error_handling = execution.get('error_handling', {})
        retry_attempts = error_handling.get('retry_attempts')
        if retry_attempts is not None:
            if not isinstance(retry_attempts, int) or retry_attempts < 0:
                self.errors.append("execution.error_handling.retry_attempts must be a non-negative integer")
    
    def _validate_algorithm_model_availability(self, config: Dict[str, Any]):
        """Validate that trained models exist for active algorithms"""
        active_algorithms = config.get('algorithms', {}).get('active', [])
        models_dir = config.get('paths', {}).get('models_dir')
        env_id = config.get('environment', {}).get('id')
        seeds = config.get('data', {}).get('seeds', [])
        
        if not models_dir or not env_id or not seeds:
            return
        
        models_path = Path(models_dir) / env_id
        if not models_path.exists():
            self.warnings.append(f"Models directory does not exist: {models_path}")
            return
        
        for algorithm in active_algorithms:
            missing_seeds = []
            for seed in seeds:
                # Check various possible model file patterns
                model_patterns = [
                    f"seed_{seed}_*/best_model.zip",
                    f"{algorithm}_*/seed_{seed}/best_model.zip",
                    f"*/{algorithm}/seed_{seed}/best_model.zip"
                ]
                
                found = False
                for pattern in model_patterns:
                    if list(models_path.glob(pattern)):
                        found = True
                        break
                
                if not found:
                    missing_seeds.append(seed)
            
            if missing_seeds:
                self.warnings.append(f"Missing model files for {algorithm}, seeds: {missing_seeds}")
    
    def _validate_metric_algorithm_compatibility(self, config: Dict[str, Any]):
        """Validate that metrics are compatible with active algorithms"""
        active_algorithms = config.get('algorithms', {}).get('active', [])
        metrics = config.get('metrics', {})
        
        # Check for CRPS variant compatibility
        if 'distributional' in metrics and 'crps' in metrics['distributional']:
            crps_config = metrics['distributional']['crps']
            if isinstance(crps_config, dict) and 'variants' in crps_config:
                variants = crps_config['variants']
                
                if 'qrdqn' not in active_algorithms and 'quantile' in variants:
                    self.warnings.append("CRPS quantile variant enabled but QRDQN not in active algorithms")
                
                if not any(algo in active_algorithms for algo in ['bootstrapped_dqn', 'mcdropout_dqn']) and 'gaussian' in variants:
                    self.warnings.append("CRPS gaussian variant enabled but no compatible algorithms active")
    
    def _estimate_resource_requirements(self, config: Dict[str, Any]):
        """Estimate resource requirements and warn if excessive"""
        n_seeds = len(config.get('data', {}).get('seeds', []))
        n_episodes = config.get('data', {}).get('n_episodes_per_seed', 10)
        n_algorithms = len(config.get('algorithms', {}).get('active', []))
        noise_levels = config.get('environment', {}).get('noise_levels', {})
        active_noise_count = len(config.get('environment', {}).get('active_noise_levels', noise_levels.keys()))
        
        total_experiments = n_seeds * n_algorithms * active_noise_count
        total_episodes = total_experiments * n_episodes
        
        # Storage estimation (rough)
        estimated_storage_gb = total_episodes * 0.001  # 1MB per episode estimate
        
        if total_experiments > 1000:
            self.warnings.append(f"Large number of experiments ({total_experiments}), consider reducing scope")
        
        if estimated_storage_gb > 10:
            self.warnings.append(f"Estimated storage requirement: {estimated_storage_gb:.1f}GB")
        
        # Runtime estimation
        estimated_runtime_hours = total_experiments * 0.1  # 6 minutes per experiment estimate
        if estimated_runtime_hours > 24:
            self.warnings.append(f"Estimated runtime: {estimated_runtime_hours:.1f} hours, consider parallel execution")
    
    def generate_validation_report(self, config: Dict[str, Any]) -> str:
        """
        Generate a comprehensive validation report
        
        Args:
            config: Configuration to validate
            
        Returns:
            Formatted validation report
        """
        errors, warnings = self.validate_full_config(config)
        
        report = ["Configuration Validation Report", "=" * 40]
        
        if errors:
            report.append(f"\nERRORS ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                report.append(f"  {i}. {error}")
        else:
            report.append("\nERRORS: None")
        
        if warnings:
            report.append(f"\nWARNINGS ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                report.append(f"  {i}. {warning}")
        else:
            report.append("\nWARNINGS: None")
        
        # Summary
        total_experiments = self._count_total_experiments(config)
        report.append(f"\nSUMMARY:")
        report.append(f"  Total experiments: {total_experiments}")
        report.append(f"  Active algorithms: {len(config.get('algorithms', {}).get('active', []))}")
        report.append(f"  Seeds: {len(config.get('data', {}).get('seeds', []))}")
        
        validation_status = "PASSED" if not errors else "FAILED"
        report.append(f"  Validation status: {validation_status}")
        
        return "\n".join(report)
    
    def _count_total_experiments(self, config: Dict[str, Any]) -> int:
        """Count total number of experiments"""
        n_seeds = len(config.get('data', {}).get('seeds', []))
        n_algorithms = len(config.get('algorithms', {}).get('active', []))
        noise_levels = config.get('environment', {}).get('noise_levels', {})
        active_noise_count = len(config.get('environment', {}).get('active_noise_levels', noise_levels.keys()))
        
        return n_seeds * n_algorithms * active_noise_count