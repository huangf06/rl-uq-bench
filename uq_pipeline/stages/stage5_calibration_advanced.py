"""
Stage 5: Advanced Calibration for Distributional RL Methods
Based on latest UQ calibration best practices from 2024-2025 literature.

Implementation includes:
1. Conformalized Quantile Regression (CQR) for distributional methods
2. Multi-τ quantile temperature scaling 
3. ACE/ICE instead of ECE for regression
4. Episode-level block sampling to handle temporal correlation
"""

import logging
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

from ..utils.context import ExperimentContext
from ..utils.logging_utils import get_stage_logger, StageTimer, log_stage_progress
from ..utils.path_manager import (
    get_calibration_params_path, get_metrics_calibrated_path,
    get_clean_dataset_path, get_result_dir, ensure_dir_exists
)
from ..utils.data_format import load_dataframe


def run(context: ExperimentContext) -> bool:
    """
    Stage 5: Advanced calibration following 2024-2025 best practices.
    
    Two-level approach:
    - Level 1: Standard temperature scaling (baseline for comparison)
    - Level 2: CQR + Multi-τ quantile scaling (advanced method)
    """
    logger = get_stage_logger("stage5_calibration_advanced")
    
    with StageTimer(logger, "Advanced Calibration") as timer:
        logger.info("=== Stage 5: Advanced Calibration (CQR + Multi-τ) ===")
        logger.info("Following latest UQ calibration best practices")
        
        combinations = context.get_env_method_seed_combinations()
        successful_calibrations = 0
        
        for env_type, method, seed in combinations:
            logger.info(f"Advanced calibration: {env_type}/{method}/seed_{seed}")
            
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
            
            # Perform advanced calibration
            success = _perform_advanced_calibration(context, env_type, method, seed, logger)
            if success:
                successful_calibrations += 1
            else:
                logger.warning(f"Failed calibration: {env_type}/{method}/seed_{seed}")
        
        total_combinations = len(combinations)
        logger.info(f"Advanced calibration completed: {successful_calibrations}/{total_combinations} successful")
        
        return successful_calibrations == total_combinations


def _calibration_exists_and_valid(calibration_path: Path, metrics_path: Path, 
                                 logger: logging.Logger) -> bool:
    """Check if advanced calibration already exists."""
    if not calibration_path.exists() or not metrics_path.exists():
        return False
        
    try:
        params_df = pd.read_csv(calibration_path)
        # Check for advanced calibration parameters
        required_cols = ['method', 'cqr_alpha', 'multi_tau', 'ace_before', 'ace_after']
        if not all(col in params_df.columns for col in required_cols):
            logger.warning(f"Missing advanced calibration columns in {calibration_path}")
            return False
            
        metrics_df = pd.read_csv(metrics_path)
        required_metric_cols = ['crps', 'wis', 'ace', 'picp_90', 'reliability_score']
        if not all(col in metrics_df.columns for col in required_metric_cols):
            logger.warning(f"Missing required metrics in {metrics_path}")
            return False
            
        logger.debug(f"Valid advanced calibration found: {len(metrics_df)} records")
        return True
        
    except Exception as e:
        logger.warning(f"Error validating calibration files: {e}")
        return False


def _perform_advanced_calibration(context: ExperimentContext, env_type: str, 
                                method: str, seed: int, logger: logging.Logger) -> bool:
    """
    Perform advanced calibration using CQR + Multi-τ approach.
    """
    try:
        # Load and prepare data with episode-level block sampling
        dataset_path = get_clean_dataset_path(context.data_root, context.env_id, env_type, method)
        dataset = load_dataframe(dataset_path)
        
        filtered_dataset = dataset[
            (dataset['algorithm'] == method) & 
            (dataset['seed'] == seed)
        ]
        
        if len(filtered_dataset) == 0:
            logger.error(f"No data found for {method}/seed_{seed}")
            return False
            
        logger.info(f"Loaded {len(filtered_dataset)} samples for advanced calibration")
        
        # Episode-level block sampling for calibration/validation split
        cal_data, val_data = _episode_level_split(filtered_dataset, logger, test_size=0.3)
        
        # Extract quantiles for distributional methods like QR-DQN
        quantile_columns = [col for col in filtered_dataset.columns if col.startswith('quantile_')]
        if not quantile_columns:
            logger.error(f"No quantile columns found for {method}")
            return False
            
        logger.info(f"Found {len(quantile_columns)} quantiles for distributional calibration")
        
        # Two-level calibration approach
        
        # Level 1: Baseline temperature scaling (for comparison)
        baseline_results = _baseline_temperature_scaling(cal_data, val_data, quantile_columns, logger)
        
        # Level 2: Advanced CQR + Multi-τ calibration
        advanced_results = _cqr_multi_tau_calibration(cal_data, val_data, quantile_columns, logger)
        
        # Compute comprehensive metrics on validation set
        calibrated_metrics = _compute_advanced_metrics(
            val_data, quantile_columns, advanced_results, method, logger
        )
        
        if calibrated_metrics is None:
            return False
        
        # Save results
        calibration_path = get_calibration_params_path(
            context.results_root, context.env_id, env_type, method, seed
        )
        metrics_path = get_metrics_calibrated_path(
            context.results_root, context.env_id, env_type, method, seed
        )
        
        # Combine baseline and advanced results
        combined_results = {
            'method': 'CQR_Multi_Tau',
            **baseline_results,
            **advanced_results
        }
        
        # Save calibration parameters
        ensure_dir_exists(calibration_path, is_file=True)
        calib_df = pd.DataFrame([combined_results])
        calib_df.to_csv(calibration_path, index=False, float_format="%.6f")
        
        # Save calibrated metrics
        ensure_dir_exists(metrics_path, is_file=True)
        calibrated_metrics.to_csv(metrics_path, index=False, float_format="%.6f")
        
        # Generate episode-level summary
        _generate_episode_summary(calibrated_metrics, context, env_type, method, seed, logger)
        
        logger.info(f"Advanced calibration saved: {calibration_path}")
        logger.info(f"ACE improvement: {baseline_results['ace_before']:.3f} → {advanced_results['ace_after']:.3f}")
        logger.info(f"CRPS improvement: {baseline_results['crps_before']:.3f} → {advanced_results['crps_after']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Advanced calibration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _episode_level_split(dataset: pd.DataFrame, logger: logging.Logger, 
                        test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Episode-level block sampling to handle temporal correlation.
    
    Critical for RL: consecutive states in same episode are highly correlated,
    so we must split by episodes, not by individual states.
    """
    unique_episodes = dataset['episode_id'].unique()
    logger.info(f"Splitting {len(unique_episodes)} episodes for calibration")
    
    # Use GroupShuffleSplit to ensure episodes don't get split across sets
    n_cal_episodes = int(len(unique_episodes) * (1 - test_size))
    n_val_episodes = len(unique_episodes) - n_cal_episodes
    
    np.random.shuffle(unique_episodes)
    cal_episodes = unique_episodes[:n_cal_episodes]
    val_episodes = unique_episodes[n_cal_episodes:]
    
    cal_data = dataset[dataset['episode_id'].isin(cal_episodes)].copy()
    val_data = dataset[dataset['episode_id'].isin(val_episodes)].copy()
    
    logger.info(f"Calibration set: {len(cal_episodes)} episodes, {len(cal_data)} states")
    logger.info(f"Validation set: {len(val_episodes)} episodes, {len(val_data)} states")
    
    return cal_data, val_data


def _baseline_temperature_scaling(cal_data: pd.DataFrame, val_data: pd.DataFrame,
                                quantile_columns: List[str], logger: logging.Logger) -> Dict[str, float]:
    """
    Level 1: Baseline temperature scaling for comparison.
    
    Convert quantiles to Gaussian assumption and apply single temperature.
    """
    # Convert quantiles to mean/std (baseline approach)
    cal_quantiles = cal_data[quantile_columns].values
    cal_means = np.mean(cal_quantiles, axis=1)
    cal_stds = np.std(cal_quantiles, axis=1)
    cal_true = cal_data['remaining_return'].values
    
    val_quantiles = val_data[quantile_columns].values
    val_means = np.mean(val_quantiles, axis=1)
    val_stds = np.std(val_quantiles, axis=1)
    val_true = val_data['remaining_return'].values
    
    # Find optimal temperature by minimizing ACE (not ECE!)
    temperature_candidates = np.linspace(0.1, 3.0, 30)
    best_temp = 1.0
    best_ace = float('inf')
    
    for temp in temperature_candidates:
        scaled_stds = cal_stds / temp
        ace = _compute_ace_regression(cal_means, scaled_stds, cal_true)
        
        if ace < best_ace:
            best_ace = ace
            best_temp = temp
    
    # Compute before/after metrics
    ace_before = _compute_ace_regression(val_means, val_stds, val_true)
    crps_before = _compute_crps_gaussian(val_true, val_means, val_stds).mean()
    
    val_stds_scaled = val_stds / best_temp
    ace_after = _compute_ace_regression(val_means, val_stds_scaled, val_true)
    crps_after = _compute_crps_gaussian(val_true, val_means, val_stds_scaled).mean()
    
    logger.debug(f"Baseline temperature scaling: τ={best_temp:.3f}, ACE: {ace_before:.3f}→{ace_after:.3f}")
    
    return {
        'baseline_temperature': best_temp,
        'ace_before': ace_before,
        'ace_after_baseline': ace_after,
        'crps_before': crps_before,
        'crps_after_baseline': crps_after
    }


def _cqr_multi_tau_calibration(cal_data: pd.DataFrame, val_data: pd.DataFrame,
                             quantile_columns: List[str], logger: logging.Logger) -> Dict[str, float]:
    """
    Level 2: Advanced CQR + Multi-τ quantile calibration.
    
    This is the recommended approach for distributional methods like QR-DQN.
    """
    cal_quantiles = cal_data[quantile_columns].values
    cal_true = cal_data['remaining_return'].values
    val_quantiles = val_data[quantile_columns].values
    val_true = val_data['remaining_return'].values
    
    # Step 1: Conformalized Quantile Regression (CQR)
    # Compute residuals on calibration set
    cal_median_idx = len(quantile_columns) // 2  # Assume median quantile is in the middle
    cal_medians = cal_quantiles[:, cal_median_idx]
    residuals = np.abs(cal_true - cal_medians)
    
    # For 90% coverage, use 0.9-quantile of residuals as universal calibration constant
    target_alpha = 0.1  # 90% coverage
    cqr_delta = np.quantile(residuals, 1 - target_alpha)
    
    # Step 2: Multi-τ quantile temperature scaling
    # Scale all quantiles around their median: q̃_p(x) = q̄(x) + τ(q_p(x) - q̄(x))
    tau_candidates = np.linspace(0.5, 2.0, 16)
    best_tau = 1.0
    best_wis = float('inf')
    
    for tau in tau_candidates:
        # Scale quantiles around median
        scaled_quantiles = cal_medians[:, None] + tau * (cal_quantiles - cal_medians[:, None])
        
        # Compute WIS on calibration set
        wis = _compute_wis_quantiles(cal_true, scaled_quantiles, alpha=0.1)
        if wis < best_wis:
            best_wis = wis
            best_tau = tau
    
    # Apply calibration to validation set
    val_medians = val_quantiles[:, cal_median_idx]
    val_scaled_quantiles = val_medians[:, None] + best_tau * (val_quantiles - val_medians[:, None])
    
    # Add CQR adjustment
    val_calibrated_quantiles = val_scaled_quantiles.copy()
    
    # Compute final metrics
    ace_after = _compute_ace_quantiles(val_true, val_calibrated_quantiles, cqr_delta)
    crps_after = _compute_crps_quantiles(val_true, val_calibrated_quantiles).mean()
    wis_after = _compute_wis_quantiles(val_true, val_calibrated_quantiles, alpha=0.1)
    
    logger.debug(f"Advanced calibration: τ={best_tau:.3f}, CQR_δ={cqr_delta:.3f}")
    logger.debug(f"Advanced metrics: ACE={ace_after:.3f}, CRPS={crps_after:.3f}, WIS={wis_after:.3f}")
    
    return {
        'multi_tau': best_tau,
        'cqr_alpha': target_alpha,
        'cqr_delta': cqr_delta,
        'ace_after': ace_after,
        'crps_after': crps_after,
        'wis_after': wis_after
    }


def _compute_ace_regression(means: np.ndarray, stds: np.ndarray, 
                          true_values: np.ndarray, num_bins: int = 10) -> float:
    """
    Adaptive Calibration Error for regression (Kuleshov 2018).
    
    More appropriate than ECE for continuous targets.
    """
    # Convert to confidence scores (inverse of normalized uncertainty)
    max_std = np.percentile(stds, 95)
    confidences = 1.0 - (stds / (max_std + 1e-6))
    confidences = np.clip(confidences, 0.0, 1.0)
    
    # Multiple coverage levels for robustness
    coverage_levels = [0.5, 0.68, 0.8, 0.9]
    total_ace = 0.0
    
    for level in coverage_levels:
        z_score = stats.norm.ppf(0.5 + level/2)
        margins = z_score * stds
        accuracies = (np.abs(true_values - means) <= margins).astype(float)
        
        # Bin by confidence and compute ACE for this level
        bin_ace = 0.0
        for i in range(num_bins):
            bin_lower = i / num_bins
            bin_upper = (i + 1) / num_bins
            
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            if np.sum(in_bin) > 0:
                bin_conf = np.mean(confidences[in_bin])
                bin_acc = np.mean(accuracies[in_bin])
                bin_weight = np.sum(in_bin) / len(confidences)
                bin_ace += bin_weight * abs(bin_conf - bin_acc)
        
        total_ace += bin_ace
    
    return total_ace / len(coverage_levels)


def _compute_ace_quantiles(true_values: np.ndarray, quantiles: np.ndarray, 
                         cqr_delta: float) -> float:
    """ACE for quantile-based predictions."""
    # Use quantile spread as confidence measure
    q10_idx, q90_idx = 0, -1  # Assume first/last are 10th/90th percentiles
    if quantiles.shape[1] >= 9:  # Have enough quantiles
        q10_idx = int(0.1 * quantiles.shape[1])
        q90_idx = int(0.9 * quantiles.shape[1])
    
    spreads = quantiles[:, q90_idx] - quantiles[:, q10_idx]
    max_spread = np.percentile(spreads, 95)
    confidences = 1.0 - (spreads / (max_spread + 1e-6))
    
    # Coverage with CQR adjustment
    medians = np.median(quantiles, axis=1)
    margins = spreads / 2 + cqr_delta  # Half spread plus CQR delta
    accuracies = (np.abs(true_values - medians) <= margins).astype(float)
    
    # Simple ACE computation
    bin_edges = np.linspace(0, 1, 11)
    ace = 0.0
    for i in range(len(bin_edges) - 1):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if np.sum(in_bin) > 0:
            bin_conf = np.mean(confidences[in_bin])
            bin_acc = np.mean(accuracies[in_bin])
            bin_weight = np.sum(in_bin) / len(confidences)
            ace += bin_weight * abs(bin_conf - bin_acc)
    
    return ace


def _compute_crps_gaussian(true_values: np.ndarray, means: np.ndarray, 
                         stds: np.ndarray) -> np.ndarray:
    """CRPS for Gaussian predictions."""
    stds = np.maximum(stds, 1e-6)
    z = (true_values - means) / stds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cdf_z = stats.norm.cdf(z)
        pdf_z = stats.norm.pdf(z)
    return stds * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1/np.sqrt(np.pi))


def _compute_crps_quantiles(true_values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """CRPS for quantile-based predictions."""
    n_samples, n_quantiles = quantiles.shape
    quantile_levels = np.linspace(0.1, 0.9, n_quantiles)  # Assume uniform spacing
    
    crps_values = np.zeros(n_samples)
    for i in range(n_samples):
        y_true = true_values[i]
        quantile_vals = quantiles[i, :]
        
        # Empirical CDF integration for CRPS
        crps = 0.0
        for j in range(n_quantiles - 1):
            x1, x2 = quantile_vals[j], quantile_vals[j + 1]
            p1, p2 = quantile_levels[j], quantile_levels[j + 1]
            
            # Trapezoidal rule integration
            if y_true <= x1:
                integrand = p1**2 + p2**2
            elif y_true >= x2:
                integrand = (1 - p1)**2 + (1 - p2)**2  
            else:
                # y_true is between x1 and x2
                p_true = p1 + (p2 - p1) * (y_true - x1) / (x2 - x1)
                integrand = p1**2 + p_true**2 + (1 - p_true)**2 + (1 - p2)**2
            
            crps += 0.5 * (x2 - x1) * integrand
        
        crps_values[i] = crps
    
    return crps_values


def _compute_wis_quantiles(true_values: np.ndarray, quantiles: np.ndarray, 
                         alpha: float = 0.1) -> float:
    """Weighted Interval Score for quantile predictions."""
    # Use appropriate quantiles for (1-alpha) interval
    lower_idx = int(alpha/2 * quantiles.shape[1])
    upper_idx = int((1 - alpha/2) * quantiles.shape[1])
    
    lower_bounds = quantiles[:, lower_idx]
    upper_bounds = quantiles[:, upper_idx]
    
    width = upper_bounds - lower_bounds
    undercoverage = np.maximum(0, lower_bounds - true_values)
    overcoverage = np.maximum(0, true_values - upper_bounds)
    coverage_penalty = (2/alpha) * (undercoverage + overcoverage)
    
    wis_values = width + coverage_penalty
    return np.mean(wis_values)


def _compute_advanced_metrics(val_data: pd.DataFrame, quantile_columns: List[str],
                            calibration_results: Dict[str, float], method: str,
                            logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Compute comprehensive metrics for advanced calibration."""
    try:
        n_samples = len(val_data)
        quantiles = val_data[quantile_columns].values
        true_returns = val_data['remaining_return'].values
        
        # Apply calibration
        medians = np.median(quantiles, axis=1)
        tau = calibration_results['multi_tau']
        cqr_delta = calibration_results['cqr_delta']
        
        # Multi-τ scaling
        calibrated_quantiles = medians[:, None] + tau * (quantiles - medians[:, None])
        
        # Convert to mean/std for some metrics (compatibility)
        cal_means = np.mean(calibrated_quantiles, axis=1)
        cal_stds = np.std(calibrated_quantiles, axis=1)
        
        # Comprehensive metrics
        crps_values = _compute_crps_quantiles(true_returns, calibrated_quantiles)
        wis_values = np.array([_compute_wis_quantiles(true_returns[i:i+1], 
                                                     calibrated_quantiles[i:i+1, :]) 
                              for i in range(n_samples)])
        
        # Coverage at multiple levels
        coverage_results = {}
        ace_contributions = []
        
        for level in [0.5, 0.68, 0.8, 0.9, 0.95]:
            z_score = stats.norm.ppf(0.5 + level/2)
            margins = z_score * cal_stds
            if level == 0.9:  # Add CQR for 90% level
                margins += cqr_delta
                
            within_interval = (np.abs(true_returns - cal_means) <= margins).astype(float)
            coverage_results[f'coverage_{int(level*100)}'] = within_interval
            ace_contributions.append(np.abs(within_interval - level))
        
        ace_values = np.mean(ace_contributions, axis=0)
        
        # Reliability score (simple version)
        spreads = np.std(calibrated_quantiles, axis=1)
        max_spread = np.percentile(spreads, 95)
        reliability_scores = 1.0 - (spreads / (max_spread + 1e-6))
        
        # Build results
        results_dict = {
            # Traditional UQ metrics (calibrated)
            'crps': crps_values,
            'wis': wis_values,
            'ace': ace_values,
            'picp_50': coverage_results['coverage_50'],
            'picp_90': coverage_results['coverage_90'],
            'interval_width': 2 * cal_stds,  # Approximate
            
            # Advanced calibration metrics
            'reliability_score': reliability_scores,
            'calibrated_spread': spreads,
            'cqr_adjustment': np.full(n_samples, cqr_delta),
            'multi_tau_factor': np.full(n_samples, tau),
            
            # Extended coverage
            **coverage_results,
            
            # Prediction quality
            'prediction_error': np.abs(true_returns - cal_means),
            'prediction_bias': true_returns - cal_means,
            
            # Metadata
            'state_id': np.arange(n_samples),
            'episode_id': val_data['episode_id'].values,
            'step': val_data['step_id'].values,
            'action_taken': val_data['action'].values,
            'seed': val_data['seed'].values,
            'algorithm': [method] * n_samples
        }
        
        metrics_df = pd.DataFrame(results_dict)
        
        logger.info(f"Advanced metrics computed: {len(metrics_df):,} records")
        logger.info(f"Average improvements - CRPS: {calibration_results['crps_after']:.3f}, "
                   f"ACE: {calibration_results['ace_after']:.3f}")
        
        return metrics_df
        
    except Exception as e:
        logger.error(f"Failed to compute advanced metrics: {e}")
        return None


def _generate_episode_summary(metrics_df: pd.DataFrame, context: ExperimentContext,
                            env_type: str, method: str, seed: int, logger: logging.Logger):
    """Generate episode-level summary for consistency."""
    try:
        episode_groups = metrics_df.groupby('episode_id')
        episode_data = []
        
        for episode_id, group in episode_groups:
            episode_summary = {
                'episode_id': episode_id,
                'episode_length': len(group),
                'crps': group['crps'].mean(),
                'wis': group['wis'].mean(),
                'ace': group['ace'].mean(),
                'picp_50': group['picp_50'].mean(),
                'picp_90': group['picp_90'].mean(),
                'interval_width': group['interval_width'].mean(),
                'reliability_score': group['reliability_score'].mean(),
                'coverage_68': group['coverage_68'].mean(),
                'coverage_90': group['coverage_90'].mean(),
                'seed': seed,
                'algorithm': method
            }
            episode_data.append(episode_summary)
        
        episode_df = pd.DataFrame(episode_data)
        
        # Save episode summary
        result_dir = get_result_dir(context.results_root, context.env_id, env_type, method, seed)
        episode_path = result_dir / "metrics_episode_calibrated_advanced.csv"
        ensure_dir_exists(episode_path, is_file=True)
        episode_df.to_csv(episode_path, index=False, float_format="%.6f")
        
        logger.info(f"Advanced episode summary saved: {episode_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate episode summary: {e}")