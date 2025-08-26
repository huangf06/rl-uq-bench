"""
Stage 4a: Episode-Level UQ Metrics (Fast Path)

Implements the recommended two-tier UQ evaluation architecture:
- Episode-level aggregation for main evaluation
- Core metrics only (6 indicators)
- Follows 2024-2025 UQ-RL best practices
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import warnings
from ..utils.context import ExperimentContext
from ..utils.logging_utils import get_stage_logger, StageTimer, log_stage_progress
from ..utils.path_manager import (
    get_metrics_episode_path, get_q_values_path, get_clean_dataset_path,
    ensure_dir_exists
)
from ..utils.data_format import (
    load_q_values, load_dataframe
)


def run(context: ExperimentContext) -> bool:
    """
    Stage 4a: Compute episode-level UQ metrics (fast path).
    
    This implements the recommended approach from recent UQ-RL literature:
    - Episode-level aggregation instead of state-action level
    - 6 core metrics instead of 23+ indicators
    - Vectorized computation for efficiency
    
    Args:
        context: Experiment context with configuration
        
    Returns:
        True if all metric computations completed successfully
    """
    logger = get_stage_logger("stage4a_episode")
    
    with StageTimer(logger, "Episode-Level UQ Metrics") as timer:
        logger.info("=== Stage 4a: Episode-Level UQ Metrics ===")
        logger.info("Following 2024-2025 UQ-RL best practices")
        
        combinations = context.get_env_method_seed_combinations()
        total_combinations = len(combinations)
        
        success_count = 0
        
        for i, (env_type, method, seed) in enumerate(combinations, 1):
            log_stage_progress(logger, "Episode Metrics", i, total_combinations, "combinations")
            
            # Check if episode metrics already computed
            episode_metrics_path = get_metrics_episode_path(
                context.results_root, context.env_id, env_type, method, seed
            )
            
            if _episode_metrics_exist_and_valid(episode_metrics_path, logger):
                logger.info(f"Episode metrics already computed: {env_type}/{method}/seed_{seed}")
                success_count += 1
                continue
            
            # Run episode-level metrics computation
            success = _compute_episode_metrics(context, env_type, method, seed, logger)
            
            if success:
                success_count += 1
            else:
                logger.warning(f"Failed episode metrics computation: {env_type}/{method}/seed_{seed}")
        
        logger.info(f"Episode metrics computation completed: {success_count}/{total_combinations} successful")
        return success_count == total_combinations


def _episode_metrics_exist_and_valid(episode_metrics_path: Path, logger: logging.Logger) -> bool:
    """Check if episode metrics already exist and are valid."""
    if not episode_metrics_path.exists():
        return False
        
    try:
        df = pd.read_csv(episode_metrics_path)
        
        # Check for 6 core metrics
        required_columns = ['episode_id', 'crps', 'wis', 'ace', 'picp_50', 'picp_90', 'interval_width']
        
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns in {episode_metrics_path}")
            return False
            
        if len(df) == 0:
            logger.warning(f"Empty episode metrics file: {episode_metrics_path}")
            return False
            
        # Check for NaN values in core metrics
        if df[['crps', 'wis', 'ace', 'interval_width']].isna().any().any():
            logger.warning(f"NaN values found in episode metrics: {episode_metrics_path}")
            return False
            
        logger.debug(f"Valid episode metrics file found: {episode_metrics_path} ({len(df)} episodes)")
        return True
        
    except Exception as e:
        logger.warning(f"Error validating episode metrics file {episode_metrics_path}: {e}")
        return False


def _compute_episode_metrics(context: ExperimentContext, env_type: str, 
                           method: str, seed: int, logger: logging.Logger) -> bool:
    """
    Compute episode-level UQ metrics for specific combination.
    
    Follows recommended practices:
    - Aggregate state-action pairs to episode level
    - Compute 6 core metrics only
    - Use vectorized operations for efficiency
    """
    logger.info(f"Computing episode UQ metrics: {env_type}/{method}/seed_{seed}")
    
    try:
        # Load evaluation dataset from Stage 1
        dataset_path = get_clean_dataset_path(context.data_root, context.env_id, env_type, method)
        
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return False
            
        dataset = load_dataframe(dataset_path)
        
        # Filter dataset to match algorithm and seed
        filtered_dataset = dataset[
            (dataset['algorithm'] == method) & 
            (dataset['seed'] == seed)
        ]
        
        if len(filtered_dataset) == 0:
            logger.error(f"No data found for {method}/seed_{seed} in dataset")
            return False
            
        logger.info(f"Loaded dataset: {len(filtered_dataset)} state-action pairs")
        
        # Compute episode-level UQ metrics
        episode_metrics_df = _calculate_episode_uq_metrics(
            filtered_dataset, method, logger
        )
        
        if episode_metrics_df is None or len(episode_metrics_df) == 0:
            logger.error("Failed to compute episode UQ metrics")
            return False
        
        # Save episode metrics
        episode_metrics_path = get_metrics_episode_path(
            context.results_root, context.env_id, env_type, method, seed
        )
        ensure_dir_exists(episode_metrics_path, is_file=True)
        
        episode_metrics_df.to_csv(episode_metrics_path, index=False, float_format="%.6f")
        
        logger.info(f"Episode metrics saved: {episode_metrics_path} ({len(episode_metrics_df)} episodes)")
        return True
        
    except Exception as e:
        logger.error(f"Episode metrics computation failed: {e}")
        return False


def _calculate_episode_uq_metrics(filtered_dataset: pd.DataFrame, method: str, 
                                logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Calculate episode-level UQ metrics using vectorized operations.
    
    Core metrics (6 indicators):
    1. CRPS - Proper score for distributional accuracy
    2. WIS - Weighted interval score  
    3. ACE - Average calibration error
    4. PICP@50 - Prediction interval coverage probability (50%)
    5. PICP@90 - Prediction interval coverage probability (90%)
    6. Interval Width - Average prediction interval width
    """
    try:
        logger.info(f"Computing episode-level UQ metrics for {method}")
        
        # Extract quantile columns
        quantile_columns = [col for col in filtered_dataset.columns if col.startswith('quantile_')]
        
        if len(quantile_columns) == 0:
            logger.error("No quantile columns found in dataset")
            return None
            
        # Verify remaining_return column exists (from fixed Stage 1)
        if 'remaining_return' not in filtered_dataset.columns:
            logger.error("Dataset missing 'remaining_return' column")
            return None
        
        logger.info(f"Found {len(quantile_columns)} quantile columns")
        
        # Group by episode for episode-level aggregation
        episode_metrics = []
        
        unique_episodes = filtered_dataset['episode_id'].unique()
        logger.info(f"Processing {len(unique_episodes)} episodes")
        
        for episode_id in unique_episodes:
            episode_data = filtered_dataset[filtered_dataset['episode_id'] == episode_id]
            
            if len(episode_data) == 0:
                continue
                
            # Compute episode-level metrics using vectorized operations
            episode_metrics_dict = _compute_episode_core_metrics(
                episode_data, quantile_columns, episode_id
            )
            
            episode_metrics.append(episode_metrics_dict)
        
        # Convert to DataFrame
        episode_metrics_df = pd.DataFrame(episode_metrics)
        logger.info(f"Computed episode metrics for {len(episode_metrics_df)} episodes")
        
        return episode_metrics_df
        
    except Exception as e:
        logger.error(f"Failed to calculate episode UQ metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _compute_episode_core_metrics(episode_data: pd.DataFrame, quantile_columns: List[str], 
                                episode_id: int) -> Dict[str, float]:
    """
    Compute the 6 core UQ metrics for a single episode.
    
    Aggregates all state-action pairs in the episode to episode-level metrics.
    """
    # Extract quantile matrix and ground truth values
    quantiles_matrix = episode_data[quantile_columns].values
    remaining_returns = episode_data['remaining_return'].values
    
    # Compute basic statistics
    q_means = np.mean(quantiles_matrix, axis=1)
    q_stds = np.std(quantiles_matrix, axis=1)
    
    # Ensure minimum uncertainty
    q_stds = np.maximum(q_stds, 1e-6)
    
    # 1. CRPS (Continuous Ranked Probability Score)
    crps_values = _compute_crps_vectorized(remaining_returns, q_means, q_stds)
    episode_crps = np.mean(crps_values)
    
    # 2. WIS (Weighted Interval Score) for 90% CI
    wis_values = _compute_wis_vectorized(remaining_returns, q_means, q_stds, alpha=0.1)
    episode_wis = np.mean(wis_values)
    
    # 3. ACE (Average Calibration Error)
    ace_value = _compute_ace_vectorized(remaining_returns, q_means, q_stds)
    
    # 4. PICP@50 (50% Prediction Interval Coverage)
    picp_50 = _compute_picp_vectorized(remaining_returns, q_means, q_stds, confidence=0.5)
    
    # 5. PICP@90 (90% Prediction Interval Coverage) 
    picp_90 = _compute_picp_vectorized(remaining_returns, q_means, q_stds, confidence=0.9)
    
    # 6. Average Interval Width (90% CI)
    interval_widths = _compute_interval_width_vectorized(q_stds, confidence=0.9)
    avg_interval_width = np.mean(interval_widths)
    
    return {
        'episode_id': int(episode_id),
        'episode_length': len(episode_data),
        'crps': episode_crps,
        'wis': episode_wis,
        'ace': ace_value,
        'picp_50': picp_50,
        'picp_90': picp_90,
        'interval_width': avg_interval_width,
        # Additional metadata
        'seed': int(episode_data['seed'].iloc[0]),
        'algorithm': episode_data['algorithm'].iloc[0]
    }


# Vectorized computation functions for core metrics
def _compute_crps_vectorized(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Vectorized CRPS computation for Gaussian distributions."""
    sigma = np.maximum(sigma, 1e-6)
    z = (y_true - mu) / sigma
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cdf_z = stats.norm.cdf(z)
        pdf_z = stats.norm.pdf(z)
    
    crps = sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1/np.sqrt(np.pi))
    return np.nan_to_num(crps, nan=0.0, posinf=1000.0, neginf=0.0)


def _compute_wis_vectorized(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Vectorized WIS computation."""
    sigma = np.maximum(sigma, 1e-6)
    z_score = stats.norm.ppf(1 - alpha/2)
    
    lower = mu - z_score * sigma
    upper = mu + z_score * sigma
    
    width = upper - lower
    undercoverage = np.maximum(0, lower - y_true)
    overcoverage = np.maximum(0, y_true - upper)
    coverage_penalty = (2/alpha) * (undercoverage + overcoverage)
    
    return width + coverage_penalty


def _compute_ace_vectorized(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_bins: int = 15) -> float:
    """Vectorized ACE computation with fixed number of bins."""
    # Create confidence bins
    confidence_levels = np.linspace(0.1, 0.9, n_bins)
    calibration_errors = []
    
    for conf_level in confidence_levels:
        z = stats.norm.ppf(0.5 + conf_level/2)
        margin = z * sigma
        within_interval = np.abs(y_true - mu) <= margin
        actual_coverage = np.mean(within_interval)
        calibration_errors.append(abs(actual_coverage - conf_level))
    
    return np.mean(calibration_errors)


def _compute_picp_vectorized(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, confidence: float) -> float:
    """Vectorized PICP computation."""
    z = stats.norm.ppf(0.5 + confidence/2)
    margin = z * sigma
    within_interval = np.abs(y_true - mu) <= margin
    return np.mean(within_interval)


def _compute_interval_width_vectorized(sigma: np.ndarray, confidence: float = 0.9) -> np.ndarray:
    """Vectorized interval width computation."""
    z = stats.norm.ppf(0.5 + confidence/2)
    return 2 * z * sigma


def get_core_metrics() -> List[str]:
    """Get list of 6 core UQ metrics."""
    return [
        'crps',           # Proper score
        'wis',            # Weighted interval score  
        'ace',            # Average calibration error
        'picp_50',        # 50% coverage
        'picp_90',        # 90% coverage
        'interval_width'  # Average interval width
    ]