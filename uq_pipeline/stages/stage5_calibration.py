"""
Stage 5: Two-Parameter Calibration Implementation (Δ + τ)
Based on o3's recommendation for calibration system redesign.

This module implements the two-parameter calibration approach:
1. Δ (delta): Mean bias correction - delta = mean(true_return - predicted_mean)
2. τ (tau): Variance scaling - minimizes ACE in range [0.2, 5]

Key features:
- Validation set preparation with episode-level sampling
- Grid search + Nelder-Mead optimization for τ
- QR-DQN special handling (quantiles transformation)
- Before/after calibration metrics comparison
- Results summary with (algorithm, noise, calibrated, crps, wis, ace, picp_90, delta, tau)
"""

import logging
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
from scipy.optimize import minimize_scalar
import random

from ..utils.context import ExperimentContext
from ..utils.logging_utils import get_stage_logger, StageTimer, log_stage_progress
from ..utils.path_manager import (
    get_calibration_params_path, get_metrics_calibrated_path,
    get_clean_dataset_path, ensure_dir_exists
)
from ..utils.data_format import (
    load_dataframe
)


def run(context: ExperimentContext) -> bool:
    """
    Stage 5: Two-Parameter Calibration (Δ + τ)
    
    Implements the industry-standard two-parameter calibration approach:
    1. Bias correction (Δ): Fix systematic over/under-estimation
    2. Variance scaling (τ): Optimize uncertainty calibration via ACE minimization
    
    Args:
        context: Experiment context with configuration
        
    Returns:
        True if calibration completed successfully
    """
    logger = get_stage_logger("stage5_calibration")
    
    with StageTimer(logger, "Two-Parameter Calibration") as timer:
        logger.info("=== Stage 5: Two-Parameter Calibration (Δ + τ) ===")
        logger.info("Following o3's recommendation for bias+variance calibration")
        
        combinations = context.get_env_method_seed_combinations()
        successful_calibrations = 0
        
        for env_type, method, seed in combinations:
            logger.info(f"Calibrating: {env_type}/{method}/seed_{seed}")
            
            # Check if calibration already exists
            calibration_path = get_calibration_params_path(
                context.results_root, context.env_id, env_type, method, seed
            )
            metrics_path = get_metrics_calibrated_path(
                context.results_root, context.env_id, env_type, method, seed  
            )
            
            if _calibration_exists_and_valid(calibration_path, metrics_path, logger):
                logger.info(f"Calibration already completed: {env_type}/{method}/seed_{seed}")
                successful_calibrations += 1
                continue
            
            # Perform two-parameter calibration
            success = _perform_two_parameter_calibration(context, env_type, method, seed, logger)
            if success:
                successful_calibrations += 1
            else:
                logger.warning(f"Failed calibration: {env_type}/{method}/seed_{seed}")
        
        total_combinations = len(combinations)
        logger.info(f"Calibration completed: {successful_calibrations}/{total_combinations} successful")
        
        return successful_calibrations == total_combinations

def run_legacy(context: ExperimentContext) -> bool:
    """
    Stage 5: Perform uncertainty calibration and compute adjusted metrics.
    
    This stage:
    1. Iterates through all (env_type, method, seed) combinations
    2. Loads raw metrics and Q-value distributions
    3. Performs uncertainty calibration using various methods
    4. Computes calibrated metrics
    5. Saves calibration parameters and calibrated metrics
    6. Supports resumption (skips completed calibrations)
    
    Args:
        context: Experiment context with configuration
        
    Returns:
        True if all calibrations completed successfully
    """
    logger = get_stage_logger("stage5_calibration")
    
    with StageTimer(logger, "Calibration and Adjusted Metrics") as timer:
        logger.info("=== Stage 5: Calibration and Adjusted Metrics ===")
        
        combinations = context.get_env_method_seed_combinations()
        total_combinations = len(combinations)
        
        success_count = 0
        
        for i, (env_type, method, seed) in enumerate(combinations, 1):
            log_stage_progress(logger, "Calibration", i, total_combinations, "combinations")
            
            # TODO: Check if calibration already completed
            calibration_path = get_calibration_params_path(
                context.results_root, context.env_id, env_type, method, seed
            )
            calibrated_metrics_path = get_metrics_calibrated_path(
                context.results_root, context.env_id, env_type, method, seed
            )
            
            if _calibration_exists_and_valid(calibration_path, calibrated_metrics_path, logger):
                logger.info(f"Calibration already completed: {env_type}/{method}/seed_{seed}")
                success_count += 1
                continue
            
            # TODO: Run calibration
            success = _perform_calibration(context, env_type, method, seed, logger)
            
            if success:
                success_count += 1
            else:
                logger.warning(f"Failed calibration: {env_type}/{method}/seed_{seed}")
        
        logger.info(f"Calibration completed: {success_count}/{total_combinations} successful")
        return success_count == total_combinations


def _calibration_exists_and_valid(calibration_path: Path, metrics_path: Path, 
                                 logger: logging.Logger) -> bool:
    """Check if two-parameter calibration already exists and is valid."""
    if not calibration_path.exists() or not metrics_path.exists():
        return False
        
    try:
        # Validate calibration parameters file
        params_df = pd.read_csv(calibration_path)
        required_param_cols = ['delta', 'tau', 'ace_before', 'ace_after']
        if not all(col in params_df.columns for col in required_param_cols):
            logger.warning(f"Missing required columns in {calibration_path}")
            return False
            
        # Validate calibrated metrics file  
        metrics_df = pd.read_csv(metrics_path)
        required_metric_cols = ['crps', 'wis', 'ace', 'picp_90']
        if not all(col in metrics_df.columns for col in required_metric_cols):
            logger.warning(f"Missing required columns in {metrics_path}")
            return False
            
        if len(metrics_df) == 0:
            logger.warning(f"Empty calibrated metrics file: {metrics_path}")
            return False
            
        logger.debug(f"Valid calibration files found: {len(metrics_df)} records")
        return True
        
    except Exception as e:
        logger.warning(f"Error validating calibration files: {e}")
        return False


def _perform_two_parameter_calibration(context: ExperimentContext, env_type: str, 
                                      method: str, seed: int, logger: logging.Logger) -> bool:
    """
    Perform two-parameter calibration for specific (env_type, method, seed) combination.
    
    Steps:
    1. Load and prepare validation set (episode-level sampling)
    2. Fit bias correction Δ = mean(true_return - predicted_mean)
    3. Optimize variance scaling τ to minimize ACE
    4. Apply calibration and compute metrics
    5. Save results with before/after comparison
    """
    logger.info(f"Performing two-parameter calibration: {env_type}/{method}/seed_{seed}")
    
    try:
        # Load dataset
        dataset_path = get_clean_dataset_path(context.data_root, context.env_id, env_type, method)
        dataset = load_dataframe(dataset_path)
        
        # Filter dataset for this method and seed
        filtered_dataset = dataset[
            (dataset['algorithm'] == method) & 
            (dataset['seed'] == seed)
        ]
        
        if len(filtered_dataset) == 0:
            logger.error(f"No data found for {method}/seed_{seed} in dataset")
            return False
            
        logger.info(f"Loaded {len(filtered_dataset)} samples for calibration")
        
        # Prepare validation set using episode-level sampling
        train_data, val_data = _prepare_validation_set(filtered_dataset, logger, val_ratio=0.2)
        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation samples")
        
        # Extract predictions and ground truth
        train_predictions = _extract_predictions(train_data, method, logger)
        val_predictions = _extract_predictions(val_data, method, logger)
        
        if train_predictions is None or val_predictions is None:
            logger.error("Failed to extract predictions")
            return False
        
        # Perform two-parameter calibration on training set
        calibration_params = _fit_two_parameter_calibration(
            train_predictions, method, logger
        )
        
        # Validate on validation set and compute before/after metrics
        validation_metrics = _validate_calibration(
            val_predictions, calibration_params, method, logger
        )
        
        # Apply calibration to full dataset and compute final metrics
        full_predictions = _extract_predictions(filtered_dataset, method, logger)
        calibrated_metrics = _compute_calibrated_metrics(
            filtered_dataset, full_predictions, calibration_params, method, logger
        )
        
        if calibrated_metrics is None:
            logger.error("Failed to compute calibrated metrics")
            return False
        
        # Save calibration parameters with validation results
        calibration_path = get_calibration_params_path(
            context.results_root, context.env_id, env_type, method, seed
        )
        ensure_dir_exists(calibration_path, is_file=True)
        
        # Combine calibration params with validation metrics
        calib_dict = {**calibration_params, **validation_metrics}
        calib_df = pd.DataFrame([calib_dict])
        calib_df.to_csv(calibration_path, index=False, float_format="%.6f")
        
        # Save calibrated metrics
        calibrated_metrics_path = get_metrics_calibrated_path(
            context.results_root, context.env_id, env_type, method, seed
        )
        ensure_dir_exists(calibrated_metrics_path, is_file=True)
        calibrated_metrics.to_csv(calibrated_metrics_path, index=False, float_format="%.6f")
        
        logger.info(f"Calibration parameters saved: {calibration_path}")
        logger.info(f"Calibrated metrics saved: {calibrated_metrics_path}")
        logger.info(f"Calibration: δ={calibration_params['delta']:.4f}, "
                   f"τ={calibration_params['tau']:.3f}, "
                   f"ACE: {validation_metrics.get('ace_before', 0):.3f} → {validation_metrics.get('ace_after', 0):.3f}")
        return True
        
    except Exception as e:
        logger.error(f"Two-parameter calibration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _prepare_validation_set(dataset: pd.DataFrame, logger: logging.Logger, 
                           val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare validation set using episode-level random sampling to avoid time correlation.
    
    Strategy: Sample episodes randomly, not individual states, to maintain temporal independence.
    """
    # Get unique episodes
    unique_episodes = dataset['episode_id'].unique()
    n_episodes = len(unique_episodes)
    n_val_episodes = max(1, int(n_episodes * val_ratio))
    
    # Random episode selection
    random.seed(42)  # Fixed seed for reproducibility
    val_episodes = random.sample(list(unique_episodes), n_val_episodes)
    train_episodes = [ep for ep in unique_episodes if ep not in val_episodes]
    
    # Split data by episodes
    val_data = dataset[dataset['episode_id'].isin(val_episodes)].copy()
    train_data = dataset[dataset['episode_id'].isin(train_episodes)].copy()
    
    logger.info(f"Validation set: {n_val_episodes}/{n_episodes} episodes ({len(val_data)} samples)")
    return train_data, val_data


def _extract_predictions(data: pd.DataFrame, method: str, 
                        logger: logging.Logger) -> Optional[Dict[str, np.ndarray]]:
    """Extract predictions from dataset, handling different UQ methods."""
    try:
        quantile_columns = [col for col in data.columns if col.startswith('quantile_')]
        
        if not quantile_columns:
            logger.error("No quantile columns found in dataset")
            return None
            
        quantiles_matrix = data[quantile_columns].values
        true_returns = data['remaining_return'].values
        
        if method.lower() == 'qrdqn':
            # For QR-DQN, we have direct quantiles
            q_means = np.mean(quantiles_matrix, axis=1)
            q_stds = np.std(quantiles_matrix, axis=1)
            
            # Also keep raw quantiles for special QR-DQN handling
            predictions = {
                'q_means': q_means,
                'q_stds': q_stds,
                'quantiles': quantiles_matrix,
                'true_returns': true_returns,
                'method': method
            }
        else:
            # For other methods, treat as distributional samples
            q_means = np.mean(quantiles_matrix, axis=1)
            q_stds = np.std(quantiles_matrix, axis=1)
            
            predictions = {
                'q_means': q_means,
                'q_stds': q_stds,
                'true_returns': true_returns,
                'method': method
            }
            
        return predictions
        
    except Exception as e:
        logger.error(f"Failed to extract predictions: {e}")
        return None


def _fit_two_parameter_calibration(predictions: Dict[str, np.ndarray], method: str,
                                  logger: logging.Logger) -> Dict[str, float]:
    """
    Fit two-parameter calibration: Δ (bias) + τ (variance scaling).
    
    Step 1: Δ = mean(true_return - predicted_mean)
    Step 2: τ optimization to minimize ACE via grid search + refinement
    """
    q_means = predictions['q_means']
    q_stds = predictions['q_stds']
    true_returns = predictions['true_returns']
    
    # Handle deterministic methods (e.g., DQN)
    if np.all(q_stds < 1e-6):
        q_stds = np.full_like(q_stds, 1e-6)
        logger.info(f"Using minimal uncertainty for deterministic {method}")
    
    # Step 1: Fit bias correction Δ
    delta = np.mean(true_returns - q_means)
    corrected_means = q_means + delta
    
    # Step 2: Optimize variance scaling τ to minimize ACE
    def compute_ace_for_tau(tau: float) -> float:
        """Compute ACE for given τ value."""
        calibrated_stds = q_stds * tau
        
        # Compute ACE across multiple confidence levels
        coverage_levels = [0.5, 0.8, 0.9]
        coverage_errors = []
        
        for level in coverage_levels:
            z_score = stats.norm.ppf(0.5 + level/2)
            margin = z_score * calibrated_stds
            within_interval = np.abs(true_returns - corrected_means) <= margin
            actual_coverage = np.mean(within_interval)
            coverage_error = abs(actual_coverage - level)
            coverage_errors.append(coverage_error)
        
        # ACE is average coverage error across levels
        ace = np.mean(coverage_errors)
        return ace
    
    # Grid search in range [0.1, 10] to allow both variance expansion and contraction
    tau_grid = np.linspace(0.1, 10.0, 100)  # Extended range for proper calibration
    best_tau = 1.0
    best_ace = float('inf')
    
    for tau in tau_grid:
        try:
            ace = compute_ace_for_tau(tau)
            if ace < best_ace:
                best_ace = ace
                best_tau = tau
        except:
            continue
    
    # Nelder-Mead refinement around best grid point
    try:
        result = minimize_scalar(
            compute_ace_for_tau,
            bounds=(max(0.1, best_tau - 0.5), min(10.0, best_tau + 0.5)),
            method='bounded'
        )
        if result.success and result.fun < best_ace:
            best_tau = result.x
            best_ace = result.fun
    except:
        logger.warning("Nelder-Mead refinement failed, using grid search result")
    
    # Compute calibration metrics
    ace_before = compute_ace_for_tau(1.0)  # Before calibration (τ=1)
    ace_after = best_ace
    
    calibration_params = {
        'delta': delta,
        'tau': best_tau,
        'calibration_improvement': ace_before - ace_after
    }
    
    logger.debug(f"Two-parameter calibration for {method}: "
                f"δ={delta:.4f}, τ={best_tau:.3f}, ACE: {ace_before:.3f} → {ace_after:.3f}")
    
    return calibration_params


def _validate_calibration(predictions: Dict[str, np.ndarray], 
                         calibration_params: Dict[str, float],
                         method: str, logger: logging.Logger) -> Dict[str, float]:
    """Validate calibration on validation set and compute before/after metrics."""
    q_means = predictions['q_means']
    q_stds = predictions['q_stds']
    true_returns = predictions['true_returns']
    
    delta = calibration_params['delta']
    tau = calibration_params['tau']
    
    # Apply calibration
    calibrated_means = q_means + delta
    calibrated_stds = q_stds * tau
    
    # Compute comprehensive metrics before/after
    def compute_metrics(means, stds, suffix=""):
        """Compute all relevant metrics."""
        # Handle minimal uncertainty
        stds = np.maximum(stds, 1e-6)
        
        # CRPS
        z = (true_returns - means) / stds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cdf_z = stats.norm.cdf(z)
            pdf_z = stats.norm.pdf(z)
        crps_values = stds * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1/np.sqrt(np.pi))
        crps = np.mean(np.nan_to_num(crps_values, nan=0.0, posinf=1000.0, neginf=0.0))
        
        # WIS (90% CI)
        alpha = 0.1
        z_score = stats.norm.ppf(1 - alpha/2)
        lower = means - z_score * stds
        upper = means + z_score * stds
        width = upper - lower
        undercoverage = np.maximum(0, lower - true_returns)
        overcoverage = np.maximum(0, true_returns - upper)
        coverage_penalty = (2/alpha) * (undercoverage + overcoverage)
        wis = np.mean(width + coverage_penalty)
        
        # ACE and PICP
        coverage_levels = [0.5, 0.8, 0.9]
        coverage_errors = []
        picp_dict = {}
        
        for level in coverage_levels:
            z = stats.norm.ppf(0.5 + level/2)
            margin = z * stds
            within_interval = np.abs(true_returns - means) <= margin
            actual_coverage = np.mean(within_interval)
            coverage_error = abs(actual_coverage - level)
            coverage_errors.append(coverage_error)
            picp_dict[f'picp_{int(level*100)}{suffix}'] = actual_coverage
        
        ace = np.mean(coverage_errors)
        
        return {
            f'crps{suffix}': crps,
            f'wis{suffix}': wis,
            f'ace{suffix}': ace,
            **picp_dict
        }
    
    # Before calibration (original predictions)
    before_metrics = compute_metrics(q_means, q_stds, "_before")
    
    # After calibration
    after_metrics = compute_metrics(calibrated_means, calibrated_stds, "_after")
    
    # Combine results
    validation_results = {**before_metrics, **after_metrics}
    
    # Log key improvements
    improvement_crps = (before_metrics['crps_before'] - after_metrics['crps_after']) / before_metrics['crps_before'] * 100
    improvement_ace = (before_metrics['ace_before'] - after_metrics['ace_after']) / before_metrics['ace_before'] * 100
    
    logger.info(f"Validation results - CRPS: {improvement_crps:.1f}% improvement, "
               f"ACE: {improvement_ace:.1f}% improvement")
    
    return validation_results



def _compute_calibrated_metrics(filtered_dataset: pd.DataFrame, 
                              predictions: Dict[str, np.ndarray],
                              calibration_params: Dict[str, float],
                              method: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Compute calibrated metrics for full dataset.
    
    Special handling for QR-DQN: Apply calibration to quantiles directly.
    """
    try:
        q_means = predictions['q_means']
        q_stds = predictions['q_stds']
        true_returns = predictions['true_returns']
        
        delta = calibration_params['delta']
        tau = calibration_params['tau']
        n_samples = len(filtered_dataset)
        
        # Apply two-parameter calibration
        if method.lower() == 'qrdqn' and 'quantiles' in predictions:
            # Special QR-DQN handling: calibrate quantiles directly
            quantiles = predictions['quantiles']
            q_means_orig = np.mean(quantiles, axis=1)
            
            # Apply correct two-parameter calibration: q ← δ + μ + (q - μ) * τ
            calibrated_quantiles = np.zeros_like(quantiles)
            for i in range(quantiles.shape[0]):
                for j in range(quantiles.shape[1]):
                    q = quantiles[i, j]
                    mean_q = q_means_orig[i]
                    calibrated_quantiles[i, j] = delta + mean_q + (q - mean_q) * tau
            
            # Recompute mean/std from calibrated quantiles
            calibrated_means = np.mean(calibrated_quantiles, axis=1)
            calibrated_stds = np.std(calibrated_quantiles, axis=1)
        else:
            # Standard calibration for other methods
            calibrated_means = q_means + delta
            calibrated_stds = q_stds * tau
        
        # Ensure minimal uncertainty
        calibrated_stds = np.maximum(calibrated_stds, 1e-6)
        
        # Compute all metrics
        # CRPS
        z = (true_returns - calibrated_means) / calibrated_stds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cdf_z = stats.norm.cdf(z)
            pdf_z = stats.norm.pdf(z)
        crps_values = calibrated_stds * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1/np.sqrt(np.pi))
        crps_values = np.nan_to_num(crps_values, nan=0.0, posinf=1000.0, neginf=0.0)
        
        # WIS (90% CI)
        alpha = 0.1
        z_score = stats.norm.ppf(1 - alpha/2)
        lower = calibrated_means - z_score * calibrated_stds
        upper = calibrated_means + z_score * calibrated_stds
        width = upper - lower
        undercoverage = np.maximum(0, lower - true_returns)
        overcoverage = np.maximum(0, true_returns - upper)
        coverage_penalty = (2/alpha) * (undercoverage + overcoverage)
        wis_values = width + coverage_penalty
        
        # ACE and PICP at multiple levels
        coverage_levels = [0.5, 0.8, 0.9]
        coverage_errors = []
        picp_results = {}
        
        for level in coverage_levels:
            z = stats.norm.ppf(0.5 + level/2)
            margin = z * calibrated_stds
            within_interval = np.abs(true_returns - calibrated_means) <= margin
            actual_coverage = within_interval.astype(float)
            coverage_error = np.abs(actual_coverage - level)
            
            coverage_errors.append(coverage_error)
            picp_results[f'picp_{int(level*100)}'] = actual_coverage
        
        ace_values = np.mean(coverage_errors, axis=0)
        
        # Additional quality metrics
        prediction_errors = np.abs(true_returns - calibrated_means)
        prediction_biases = true_returns - calibrated_means
        normalized_errors = prediction_errors / calibrated_stds
        
        # Build comprehensive results DataFrame with o3's recommended structure
        results_dict = {
            # Primary UQ metrics (o3's requirement: crps, wis, ace, picp_90)
            'crps': crps_values,
            'wis': wis_values,
            'ace': ace_values,
            **picp_results,
            
            # Prediction quality
            'prediction_error': prediction_errors,
            'prediction_bias': prediction_biases,
            'normalized_error': normalized_errors,
            'interval_width': width,
            'calibrated_uncertainty': calibrated_stds,
            
            # Calibration parameters (o3's requirement: delta, tau)
            'delta_applied': np.full(n_samples, delta),
            'tau_applied': np.full(n_samples, tau),
            
            # Metadata (algorithm, noise level inferred from env_type)
            'state_id': np.arange(n_samples),
            'episode_id': filtered_dataset['episode_id'].values,
            'step': filtered_dataset['step_id'].values,
            'action_taken': filtered_dataset['action'].values,
            'seed': filtered_dataset['seed'].values,
            'algorithm': [method] * n_samples,
            'calibrated': [True] * n_samples  # Flag for before/after analysis
        }
        
        calibrated_metrics = pd.DataFrame(results_dict)
        
        # Log summary statistics
        logger.info(f"Calibrated metrics computed: {len(calibrated_metrics):,} records")
        logger.info(f"Mean CRPS: {np.mean(crps_values):.3f}, "
                   f"Mean ACE: {np.mean(ace_values):.3f}, "
                   f"PICP_90: {np.mean(picp_results['picp_90']):.3f}")
        
        return calibrated_metrics
        
    except Exception as e:
        logger.error(f"Failed to compute calibrated metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None



def get_supported_calibration_methods() -> List[str]:
    """Get list of supported calibration methods."""
    return ['two_parameter_calibration']


def validate_calibration_params(params: Dict[str, Any]) -> bool:
    """Validate two-parameter calibration parameters structure."""
    required_keys = ['delta', 'tau', 'ace_before', 'ace_after']
    
    if not all(key in params for key in required_keys):
        return False
        
    # Check parameter value ranges
    if not isinstance(params['delta'], (int, float)):
        return False
    if not (0.2 <= params['tau'] <= 5.0):  # τ range as specified
        return False
    if not (0.0 <= params['ace_before'] <= 1.0):
        return False
    if not (0.0 <= params['ace_after'] <= 1.0):
        return False
        
    return True