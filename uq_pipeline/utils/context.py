"""
Experiment Context Management
Centralized configuration and experiment state management for UQ pipeline.
"""

from pathlib import Path
from typing import Dict, List, Any
import yaml


class ExperimentContext:
    """
    Central context manager for UQ experiments.
    
    Provides unified access to:
    - Configuration parameters
    - Method and seed iteration
    - Path generation
    - Experiment state tracking
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize experiment context from YAML configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    @property
    def env_id(self) -> str:
        """Get environment ID."""
        # Support both unified format and legacy format
        if 'environment' in self.config:
            return self.config['environment']['id']
        return self.config.get('env_id', 'LunarLander-v3')
    
    @property
    def uq_methods(self) -> List[str]:
        """Get list of UQ methods to evaluate (backward compatibility)."""
        # Support unified format: algorithms.active
        if 'algorithms' in self.config and isinstance(self.config['algorithms'], dict):
            return self.config['algorithms'].get('active', [])
        # Legacy format: algorithms or uq_methods
        return self.config.get('algorithms', self.config.get('uq_methods', []))
    
    @property
    def algorithms(self) -> List[str]:
        """Get list of algorithms to evaluate."""
        # Support unified format: algorithms.active
        if 'algorithms' in self.config and isinstance(self.config['algorithms'], dict):
            return self.config['algorithms'].get('active', [])
        # Legacy format: algorithms or uq_methods
        return self.config.get('algorithms', self.config.get('uq_methods', []))
    
    @property
    def env_types(self) -> List[str]:
        """Get list of environment types (backward compatibility)."""
        # Support unified format: environment.noise_levels
        if 'environment' in self.config and 'noise_levels' in self.config['environment']:
            noise_levels = self.config['environment']['noise_levels']
            return list(noise_levels.keys())
        # Legacy format: env_types_with_noise or env_types
        if 'env_types_with_noise' in self.config:
            return list(self.config['env_types_with_noise'].keys())
        return self.config.get('env_types', [])
    
    @property
    def env_types_with_noise(self) -> Dict[str, float]:
        """Get environment types with their noise levels."""
        # Support unified format: environment.noise_levels
        if 'environment' in self.config and 'noise_levels' in self.config['environment']:
            return self.config['environment']['noise_levels']
        # Legacy format
        return self.config.get('env_types_with_noise', {})
    
    @property
    def algorithm_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get algorithm-specific configurations."""
        # Support unified format: algorithms.configs
        if 'algorithms' in self.config and isinstance(self.config['algorithms'], dict):
            return self.config['algorithms'].get('configs', {})
        # Legacy format
        return self.config.get('algorithm_configs', {})
    
    @property
    def seeds(self) -> List[int]:
        """Get list of random seeds."""
        # Support unified format: data.seeds
        if 'data' in self.config and 'seeds' in self.config['data']:
            return self.config['data']['seeds']
        # Legacy format
        return self.config.get('seeds', [101, 307, 911, 1747, 2029, 2861, 3253, 4099, 7919, 9011])
    
    @property
    def n_seeds(self) -> int:
        """Get number of seeds to use."""
        return self.config.get('n_seeds', len(self.seeds))
    
    @property
    def n_episodes_per_seed(self) -> int:
        """Get number of episodes per seed."""
        return self.config.get('n_episodes_per_seed', 10)
    
    @property
    def n_mc_samples(self) -> int:
        """Get number of MC samples."""
        return self.config.get('n_mc_samples', 30)
    
    @property
    def data_root(self) -> Path:
        """Get data root directory path."""
        # Support both nested paths and direct keys
        paths = self.config.get('paths', {})
        if 'data_dir' in paths:
            data_path = Path(paths['data_dir'])
        else:
            data_path = Path(self.config.get('data_root', 'uq_results/data'))
        
        # Ensure absolute path to avoid nesting issues
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path
        return data_path
    
    @property
    def results_root(self) -> Path:
        """Get results root directory path."""
        # Support both nested paths and direct keys
        paths = self.config.get('paths', {})
        if 'results_dir' in paths:
            results_path = Path(paths['results_dir'])
        else:
            results_path = Path(self.config.get('results_root', 'uq_results/results'))
        
        # Ensure absolute path to avoid nesting issues
        if not results_path.is_absolute():
            results_path = Path.cwd() / results_path
        return results_path
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        paths = self.config.get('paths', {})
        return Path(paths.get('logs_dir', 'logs'))
    
    @property
    def eval_episodes(self) -> int:
        """Get number of evaluation episodes."""
        # Support unified format: data.eval_episodes
        if 'data' in self.config and 'eval_episodes' in self.config['data']:
            return self.config['data']['eval_episodes']
        # Legacy format
        return self.config.get('eval_episodes', 100)
    
    @property
    def success_threshold(self) -> float:
        """Get success threshold."""
        return self.config.get('success_threshold', 200)
    
    @property
    def global_seed(self) -> int:
        """Get global seed."""
        return self.config.get('global_seed', 42)
    
    @property
    def uq_metrics(self) -> List[str]:
        """Get list of UQ metrics to compute."""
        return self.config.get('uq_metrics', ['crps', 'wis', 'ece', 'rmse', 'mae', 'correlation'])
    
    @property
    def bootstrap_samples(self) -> int:
        """Get number of bootstrap samples."""
        return self.config.get('bootstrap_samples', 200)
    
    @property
    def confidence_level(self) -> float:
        """Get confidence level."""
        return self.config.get('confidence_level', 0.95)
    
    @property
    def model_load_config(self) -> Dict[str, Any]:
        """Get model loading configuration."""
        return self.config.get('model_load', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    @property
    def pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.config.get('pipeline', {})
    
    @property
    def skip_existing(self) -> bool:
        """Get whether to skip existing files."""
        return self.pipeline_config.get('skip_existing', True)
    
    def get_method_seed_combinations(self) -> List[tuple]:
        """
        Get all combinations of (method, seed) for iteration.
        
        Returns:
            List of (method_name, seed) tuples
        """
        from itertools import product
        # Use only the first n_seeds from the seeds list
        active_seeds = self.seeds[:self.n_seeds]
        return list(product(self.algorithms, active_seeds))
    
    def get_env_method_seed_combinations(self) -> List[tuple]:
        """
        Get all combinations of (env_type, method, seed) for iteration.
        
        Returns:
            List of (env_type, method_name, seed) tuples
        """
        from itertools import product
        # Use only the first n_seeds from the seeds list
        active_seeds = self.seeds[:self.n_seeds]
        return list(product(self.env_types, self.algorithms, active_seeds))
    
    def get_noise_level(self, env_type: str) -> float:
        """
        Get noise level for a specific environment type.
        
        Args:
            env_type: Environment type identifier
            
        Returns:
            Noise level as float
        """
        return self.env_types_with_noise.get(env_type, 0.0)
    
    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """
        Get configuration for a specific algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Algorithm configuration dictionary
        """
        return self.algorithm_configs.get(algorithm, {})
    
    def get_uncertainty_type(self, algorithm: str) -> str:
        """
        Get uncertainty type for a specific algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Uncertainty type string
        """
        config = self.get_algorithm_config(algorithm)
        return config.get('uncertainty_type', 'none')
    
    def get_ensemble_size(self, algorithm: str) -> int:
        """
        Get ensemble size for a specific algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Ensemble size
        """
        config = self.get_algorithm_config(algorithm)
        return config.get('ensemble_size', 1)
    
    def get_dropout_rate(self, algorithm: str) -> float:
        """
        Get dropout rate for MC Dropout algorithms.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Dropout rate
        """
        config = self.get_algorithm_config(algorithm)
        return config.get('dropout_rate', 0.05)
    
    def get_mc_samples(self, algorithm: str) -> int:
        """
        Get number of MC samples for MC Dropout algorithms.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Number of MC samples
        """
        config = self.get_algorithm_config(algorithm)
        return config.get('mc_samples', self.n_mc_samples)
    
    def get_n_quantiles(self, algorithm: str) -> int:
        """
        Get number of quantiles for quantile-based algorithms.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Number of quantiles
        """
        config = self.get_algorithm_config(algorithm)
        return config.get('n_quantiles', 170)
    
    def is_experiment_completed(self, env_type: str, method: str, seed: int) -> bool:
        """
        Check if a specific experiment combination is already completed.
        
        Args:
            env_type: Environment type identifier
            method: UQ method name
            seed: Random seed
            
        Returns:
            True if experiment is completed (summary.json exists and valid)
        """
        # TODO: Check for completed experiments using summary.json
        pass
    
    def mark_experiment_completed(self, env_type: str, method: str, seed: int) -> None:
        """
        Mark a specific experiment combination as completed.
        
        Args:
            env_type: Environment type identifier
            method: UQ method name
            seed: Random seed
        """
        # TODO: Create/update summary.json to mark completion
        pass