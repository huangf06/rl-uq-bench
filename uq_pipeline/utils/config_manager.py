"""
Unified Configuration Manager for UQ Pipeline
Handles loading, validation, and conversion of configuration files
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging


@dataclass
class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors"""
    field: str
    message: str
    
    def __str__(self):
        return f"Config validation error in '{self.field}': {self.message}"


class ConfigManager:
    """Unified configuration manager for UQ pipeline"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Field mapping for backward compatibility
        self.legacy_field_mapping = {
            'env_id': 'environment.id',
            'env_type': 'environment.current_noise_level',
            'noise_level': 'environment.current_noise_level',
            'uq_methods': 'algorithms.active',
            'algorithm_configs': 'algorithms.configs',
            'uq_metrics': 'metrics',
            'data_dir': 'paths.data_dir',
            'results_dir': 'paths.results_dir',
            'models_dir': 'paths.models_dir'
        }
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with validation and compatibility handling
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded and validated configuration dictionary
        """
        if config_path:
            self.config_path = Path(config_path)
        elif not self.config_path:
            # Check environment variable or use default
            env_config = os.getenv('UQ_CONFIG_FILE')
            if env_config:
                self.config_path = Path(env_config)
            else:
                self.config_path = Path("uq_pipeline/configs/unified_config_template.yml")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Load YAML content
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Detect configuration format and convert if needed
        if self._is_legacy_format(raw_config):
            self.logger.info("Legacy configuration format detected, converting...")
            self.config = self._convert_legacy_config(raw_config)
        else:
            self.config = raw_config
        
        # Validate configuration
        self._validate_config()
        
        # Post-process configuration
        self._post_process_config()
        
        self.logger.info(f"Configuration loaded successfully from {self.config_path}")
        return self.config
    
    def _is_legacy_format(self, config: Dict[str, Any]) -> bool:
        """
        Detect if configuration is in legacy format
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            True if legacy format detected
        """
        legacy_indicators = ['env_id', 'env_type', 'uq_methods']
        return any(key in config for key in legacy_indicators)
    
    def _convert_legacy_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy configuration format to unified format
        
        Args:
            legacy_config: Configuration in legacy format
            
        Returns:
            Configuration in unified format
        """
        unified_config = {
            'experiment': {
                'name': legacy_config.get('env_type', 'legacy_experiment'),
                'description': 'Converted from legacy configuration',
                'version': '2.0'
            },
            'environment': {
                'id': legacy_config.get('env_id', 'LunarLander-v3'),
                'success_threshold': legacy_config.get('success_threshold', 200)
            },
            'algorithms': {
                'active': legacy_config.get('algorithms', legacy_config.get('uq_methods', ['qrdqn'])),
                'configs': legacy_config.get('algorithm_configs', {})
            },
            'data': {
                'n_seeds': legacy_config.get('n_seeds', 10),
                'seeds': legacy_config.get('seeds', []),
                'n_episodes_per_seed': legacy_config.get('n_episodes_per_seed', 10),
                'n_mc_samples': legacy_config.get('n_mc_samples', 30),
                'bootstrap_samples': legacy_config.get('bootstrap_samples', 200),
                'confidence_levels': [0.5, 0.8, 0.9, 0.95]
            },
            'paths': {
                'data_dir': legacy_config.get('paths', {}).get('data_dir', 'uq_results/data'),
                'results_dir': legacy_config.get('paths', {}).get('results_dir', 'uq_results/results'),
                'models_dir': legacy_config.get('paths', {}).get('logs_dir', 'logs/multi_env_experiments')
            },
            'model_load': legacy_config.get('model_load', {}),
            'logging': legacy_config.get('logging', {}),
            'global': {
                'seed': legacy_config.get('global_seed', 42),
                'reproducible': True
            }
        }
        
        # Handle noise level configuration
        noise_level = legacy_config.get('noise_level', 0.0)
        env_type = legacy_config.get('env_type', 'clean')
        
        if 'noise' in env_type and noise_level >= 0:
            unified_config['environment']['current_noise_level'] = noise_level
        
        # Convert metrics configuration
        if 'uq_metrics' in legacy_config:
            metrics_config = {}
            for metric in legacy_config['uq_metrics']:
                if metric in ['crps', 'kl_divergence']:
                    metrics_config.setdefault('distributional', {})[metric] = {'enabled': True}
                elif metric in ['wis', 'coverage']:
                    metrics_config.setdefault('interval_based', {})[metric] = {'enabled': True}
                elif metric in ['ace', 'ece', 'picp_50', 'picp_80', 'picp_90']:
                    metrics_config.setdefault('calibration', {})[metric] = {'enabled': True}
                elif metric in ['rmse', 'mae', 'correlation']:
                    metrics_config.setdefault('performance', {})[metric] = {'enabled': True}
            
            unified_config['metrics'] = metrics_config
        
        return unified_config
    
    def _validate_config(self):
        """Validate configuration structure and values"""
        required_fields = [
            'environment.id',
            'algorithms.active',
            'data.seeds'
        ]
        
        for field_path in required_fields:
            if not self._get_nested_value(self.config, field_path):
                raise ConfigValidationError(field_path, "Required field is missing or empty")
        
        # Validate environment
        env_id = self._get_nested_value(self.config, 'environment.id')
        if not isinstance(env_id, str) or not env_id.strip():
            raise ConfigValidationError('environment.id', "Must be a non-empty string")
        
        # Validate algorithms
        active_algorithms = self._get_nested_value(self.config, 'algorithms.active')
        if not isinstance(active_algorithms, list) or not active_algorithms:
            raise ConfigValidationError('algorithms.active', "Must be a non-empty list")
        
        valid_algorithms = ['dqn', 'qrdqn', 'bootstrapped_dqn', 'mcdropout_dqn']
        for algo in active_algorithms:
            if algo not in valid_algorithms:
                raise ConfigValidationError('algorithms.active', f"Unknown algorithm '{algo}'. Valid options: {valid_algorithms}")
        
        # Validate seeds
        seeds = self._get_nested_value(self.config, 'data.seeds')
        if not isinstance(seeds, list) or not seeds:
            raise ConfigValidationError('data.seeds', "Must be a non-empty list of integers")
        
        if not all(isinstance(seed, int) for seed in seeds):
            raise ConfigValidationError('data.seeds', "All seeds must be integers")
        
        # Validate paths
        self._validate_paths()
        
        self.logger.info("Configuration validation passed")
    
    def _validate_paths(self):
        """Validate path configurations"""
        path_fields = ['data_dir', 'results_dir', 'models_dir']
        
        for field in path_fields:
            path_value = self._get_nested_value(self.config, f'paths.{field}')
            if path_value:
                path_obj = Path(path_value)
                # Check if parent directory exists for creation
                if not path_obj.parent.exists() and path_obj.parent != Path('.'):
                    self.logger.warning(f"Parent directory for {field} does not exist: {path_obj.parent}")
    
    def _post_process_config(self):
        """Post-process configuration after loading and validation"""
        # Convert path strings to Path objects
        if 'paths' in self.config:
            for key, value in self.config['paths'].items():
                if isinstance(value, str):
                    self.config['paths'][key] = Path(value)
        
        # Set up default values for missing optional fields
        self._set_defaults()
        
        # Handle lambda functions for model loading
        if 'model_load' in self.config and 'custom_objects' in self.config['model_load']:
            custom_objects = self.config['model_load']['custom_objects']
            if custom_objects.get('lr_schedule') is None:
                custom_objects['lr_schedule'] = lambda _: 0.0
            if custom_objects.get('exploration_schedule') is None:
                custom_objects['exploration_schedule'] = lambda _: 0.0
    
    def _set_defaults(self):
        """Set default values for optional configuration fields"""
        defaults = {
            'execution.pipeline.skip_existing': True,
            'execution.pipeline.cache_duration_hours': 24,
            'execution.error_handling.continue_on_failure': True,
            'execution.error_handling.retry_attempts': 2,
            'logging.level': 'INFO',
            'logging.console_enabled': True,
            'logging.file_enabled': True,
            'global.seed': 42,
            'global.reproducible': True,
            'global.compression.enabled': True,
            'global.compression.format': 'xz'
        }
        
        for field_path, default_value in defaults.items():
            if not self._get_nested_value(self.config, field_path):
                self._set_nested_value(self.config, field_path, default_value)
    
    def _get_nested_value(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], field_path: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = field_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get_experiment_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all experiment combinations based on configuration
        
        Returns:
            List of experiment combination dictionaries
        """
        combinations = []
        
        env_id = self.config['environment']['id']
        algorithms = self.config['algorithms']['active']
        seeds = self.config['data']['seeds']
        
        # Handle noise levels
        noise_levels = self.config.get('environment', {}).get('noise_levels', {'clean': 0.0})
        active_noise_levels = self.config.get('environment', {}).get('active_noise_levels', [])
        
        if not active_noise_levels:
            active_noise_levels = list(noise_levels.keys())
        
        for noise_name in active_noise_levels:
            noise_value = noise_levels.get(noise_name, 0.0)
            for algorithm in algorithms:
                for seed in seeds:
                    combinations.append({
                        'env_id': env_id,
                        'noise_name': noise_name,
                        'noise_value': noise_value,
                        'algorithm': algorithm,
                        'seed': seed
                    })
        
        return combinations
    
    def save_config(self, output_path: Union[str, Path], format: str = 'yaml'):
        """
        Save current configuration to file
        
        Args:
            output_path: Output file path
            format: Output format ('yaml' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Path objects back to strings for serialization
        serializable_config = self._make_serializable(self.config.copy())
        
        if format.lower() == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(serializable_config, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Configuration saved to {output_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert configuration object to serializable format"""
        if isinstance(obj, Path):
            return str(obj)
        elif callable(obj):
            # Handle lambda functions and other callables
            return None
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of updates using dot notation keys
        """
        for field_path, value in updates.items():
            self._set_nested_value(self.config, field_path, value)
        
        # Re-validate after updates
        self._validate_config()