"""
Path Management Utilities
Centralized path generation and validation for UQ pipeline.
"""

from pathlib import Path
from typing import Optional


def get_trained_model_path(env_id: str, env_type: str, algorithm: str, seed: int, 
                          model_filename: str = "best_model.zip") -> Path:
    """
    Generate path for trained model file.
    
    Args:
        env_id: Environment identifier (e.g., 'LunarLander-v3')
        env_type: Environment type (e.g., 'uncertainty_degradation_noise0.050')
        algorithm: Algorithm name (e.g., 'qrdqn')
        seed: Random seed
        model_filename: Model filename (default: 'best_model.zip')
        
    Returns:
        Path to trained model file
        
    Example:
        logs/multi_env_experiments/LunarLander-v3/uncertainty_degradation_noise0.050/qrdqn/seed_101_1/best_model.zip
    """
    return Path(f"logs/multi_env_experiments/{env_id}/{env_type}/{algorithm}/seed_{seed}_1/{model_filename}")


def get_trained_model_candidates(env_id: str, env_type: str, algorithm: str, seed: int) -> list[Path]:
    """
    Generate list of candidate paths for trained model files.
    
    Tries multiple common model filenames in order of preference.
    
    Args:
        env_id: Environment identifier
        env_type: Environment type
        algorithm: Algorithm name
        seed: Random seed
        
    Returns:
        List of candidate model file paths
    """
    base_dir = Path(f"logs/multi_env_experiments/{env_id}/{env_type}/{algorithm}/seed_{seed}_1")
    candidates = [
        "best_model.zip",
        "rl_model_100000_steps.zip", 
        "final_model.zip",
        f"{env_id}.zip"
    ]
    return [base_dir / filename for filename in candidates]


def get_clean_dataset_path(data_root: Path, env_id: str, env_type: str, algorithm: str = None) -> Path:
    """
    Generate path for clean evaluation dataset.
    
    Args:
        data_root: Root directory for data storage
        env_id: Environment identifier (e.g., 'LunarLander-v3')
        env_type: Environment type (e.g., 'uncertainty_degradation_noise0.050')
        algorithm: Algorithm name (e.g., 'qrdqn'). If None, uses old path structure.
        
    Returns:
        Path to eval_dataset.xz file
        
    Examples:
        # New structure (recommended):
        data_root/LunarLander-v3/uncertainty_degradation_noise0.050/qrdqn/eval_dataset.xz
        # Old structure (backward compatibility):
        data_root/LunarLander-v3/uncertainty_degradation_noise0.050/eval_dataset.xz
    """
    if algorithm:
        return data_root / env_id / env_type / algorithm / "eval_dataset.xz"
    else:
        # Backward compatibility - old path structure
        return data_root / env_id / env_type / "eval_dataset.xz"


def get_result_dir(results_root: Path, env_id: str, env_type: str, 
                   method: str, seed: int) -> Path:
    """
    Generate directory path for experiment results.
    
    Args:
        results_root: Root directory for results storage
        env_id: Environment identifier
        env_type: Environment type
        method: UQ method name
        seed: Random seed
        
    Returns:
        Path to result directory
        
    Example:
        results_root/LunarLander-v3/uncertainty_degradation_noise0.050/qrdqn/seed_101/
    """
    return results_root / env_id / env_type / method / f"seed_{seed}"


def get_performance_path(results_root: Path, env_id: str, env_type: str,
                        method: str, seed: int) -> Path:
    """
    Generate path for performance metrics file.
    
    Args:
        results_root: Root directory for results
        env_id: Environment identifier
        env_type: Environment type
        method: UQ method name
        seed: Random seed
        
    Returns:
        Path to performance.json file
    """
    result_dir = get_result_dir(results_root, env_id, env_type, method, seed)
    return result_dir / "performance.json"


def get_q_values_path(results_root: Path, env_id: str, env_type: str,
                     method: str, seed: int) -> Path:
    """
    Generate path for Q-values file.
    
    Args:
        results_root: Root directory for results
        env_id: Environment identifier
        env_type: Environment type
        method: UQ method name
        seed: Random seed
        
    Returns:
        Path to q_values.xz file
    """
    result_dir = get_result_dir(results_root, env_id, env_type, method, seed)
    return result_dir / "q_values.xz"


def get_metrics_episode_path(results_root: Path, env_id: str, env_type: str, 
                           algorithm: str, seed: int) -> Path:
    """
    Generate path for episode-level metrics file.
    
    Args:
        results_root: Root directory for results
        env_id: Environment identifier
        env_type: Environment type
        algorithm: Algorithm name  
        seed: Random seed
        
    Returns:
        Path to episode metrics CSV file
        
    Example:
        uq_results/results/LunarLander-v3/uncertainty_degradation_noise0.050/qrdqn/seed_101/metrics_episode.csv
    """
    return results_root / env_id / env_type / algorithm / f"seed_{seed}" / "metrics_episode.csv"


def get_metrics_raw_path(results_root: Path, env_id: str, env_type: str,
                        method: str, seed: int) -> Path:
    """
    Generate path for raw metrics file.
    
    Args:
        results_root: Root directory for results
        env_id: Environment identifier
        env_type: Environment type
        method: UQ method name
        seed: Random seed
        
    Returns:
        Path to metrics_raw.csv file
    """
    result_dir = get_result_dir(results_root, env_id, env_type, method, seed)
    return result_dir / "metrics_raw.csv"


def get_calibration_params_path(results_root: Path, env_id: str, env_type: str,
                               method: str, seed: int) -> Path:
    """
    Generate path for calibration parameters file.
    
    Args:
        results_root: Root directory for results
        env_id: Environment identifier
        env_type: Environment type
        method: UQ method name
        seed: Random seed
        
    Returns:
        Path to calibration_params.csv file
    """
    result_dir = get_result_dir(results_root, env_id, env_type, method, seed)
    return result_dir / "calibration_params.csv"


def get_uq_predictions_path(results_root: Path, env_id: str, env_type: str,
                           method: str, seed: int) -> Path:
    """
    Generate path for UQ predictions file (Stage 2 output).
    
    Args:
        results_root: Root directory for results
        env_id: Environment identifier
        env_type: Environment type
        method: UQ method name
        seed: Random seed
        
    Returns:
        Path to uq_predictions.xz file
    """
    result_dir = get_result_dir(results_root, env_id, env_type, method, seed)
    return result_dir / "uq_predictions.xz"


def get_metrics_calibrated_path(results_root: Path, env_id: str, env_type: str,
                               method: str, seed: int) -> Path:
    """
    Generate path for calibrated metrics file.
    
    Args:
        results_root: Root directory for results
        env_id: Environment identifier
        env_type: Environment type
        method: UQ method name
        seed: Random seed
        
    Returns:
        Path to metrics_calibrated.csv file
    """
    result_dir = get_result_dir(results_root, env_id, env_type, method, seed)
    return result_dir / "metrics_calibrated.csv"


def get_summary_path(results_root: Path, env_id: str, env_type: str,
                    method: str, seed: int) -> Path:
    """
    Generate path for experiment summary file.
    
    Args:
        results_root: Root directory for results
        env_id: Environment identifier
        env_type: Environment type
        method: UQ method name
        seed: Random seed
        
    Returns:
        Path to summary.json file
    """
    # TODO: Implement path construction for summary file
    pass


def ensure_dir_exists(path: Path, is_file: bool = True) -> None:
    """
    Ensure directory exists for given path.
    
    Args:
        path: File or directory path
        is_file: If True, create parent directory; if False, create the directory itself
    """
    if is_file:
        # Create parent directory for file
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Create the directory itself
        path.mkdir(parents=True, exist_ok=True)


def validate_required_files(result_dir: Path) -> dict:
    """
    Validate that all required output files exist in result directory.
    
    Args:
        result_dir: Path to result directory
        
    Returns:
        Dictionary with file existence status
        
    Example:
        {
            'performance.json': True,
            'q_values.xz': True,
            'metrics_raw.csv': False,
            ...
        }
    """
    # TODO: Implement file existence validation
    pass