"""
Stage 5: Standard Calibration Implementation
Based on industry best practices for uncertainty calibration.

This module implements standard temperature scaling following:
- Guo et al. 2017 "On Calibration of Modern Neural Networks"  
- Focus on Expected Calibration Error (ECE) minimization
- Simple, interpretable, and research-focused approach
"""

import logging
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy import stats

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
    Stage 5: Standard Temperature Scaling Calibration.
    
    This stage applies industry-standard calibration methods focused on research objectives:
    - Temperature scaling to minimize Expected Calibration Error (ECE)
    - No bias correction (preserve true prediction quality assessment)
    - Standard reliability diagram evaluation
    
    Args:
        context: Experiment context with configuration
        
    Returns:
        True if calibration completed successfully
    """
    logger = get_stage_logger("stage5_calibration_standard")
    
    with StageTimer(logger, "Standard Calibration") as timer:
        logger.info("=== Stage 5: Standard Calibration (Temperature Scaling) ===")
        logger.info("Following industry best practices for research evaluation")
        
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
            
            # Perform standard calibration
            success = _perform_standard_calibration(context, env_type, method, seed, logger)
            if success:
                successful_calibrations += 1
            else:
                logger.warning(f"Failed calibration: {env_type}/{method}/seed_{seed}")
        
        total_combinations = len(combinations)
        logger.info(f"Calibration completed: {successful_calibrations}/{total_combinations} successful")
        # Note: timer.elapsed not available, but timing info logged by StageTimer
        
        return successful_calibrations == total_combinations


def _calibration_exists_and_valid(calibration_path: Path, metrics_path: Path, 
                                 logger: logging.Logger) -> bool:
    """Check if calibration already exists and is valid."""
    if not calibration_path.exists() or not metrics_path.exists():
        return False
        
    try:
        # Validate calibration parameters file
        params_df = pd.read_csv(calibration_path)
        required_param_cols = ['temperature', 'ece_before', 'ece_after']
        if not all(col in params_df.columns for col in required_param_cols):
            logger.warning(f"Missing required columns in {calibration_path}")
            return False
            
        # Validate calibrated metrics file  
        metrics_df = pd.read_csv(metrics_path)
        required_metric_cols = ['confidence', 'accuracy', 'ece']
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


def _perform_standard_calibration(context: ExperimentContext, env_type: str, 
                        method: str, seed: int, logger: logging.Logger) -> bool:
    """
    Perform standard temperature scaling calibration.
    """
    logger.info(f"Performing standard calibration: {env_type}/{method}/seed_{seed}")
    
    try:
        # Load dataset to get predictions and ground truth
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
        
        # Prepare data for calibration
        quantile_columns = [col for col in filtered_dataset.columns if col.startswith('quantile_')]
        quantiles_matrix = filtered_dataset[quantile_columns].values
        q_means = np.mean(quantiles_matrix, axis=1)
        q_stds = np.std(quantiles_matrix, axis=1)
        true_returns = filtered_dataset['remaining_return'].values
        
        # Perform standard temperature scaling
        calibration_results = _standard_temperature_scaling(
            q_means, q_stds, true_returns, method, logger
        )
        
        # Compute calibrated metrics
        calibrated_metrics = _compute_standard_calibrated_metrics(
            filtered_dataset, q_means, q_stds, true_returns, 
            calibration_results, method, logger
        )
        
        if calibrated_metrics is None:
            logger.error("Failed to compute calibrated metrics")
            return False
        
        # Save calibration parameters
        calibration_path = get_calibration_params_path(
            context.results_root, context.env_id, env_type, method, seed
        )
        ensure_dir_exists(calibration_path, is_file=True)
        
        # Save as CSV for consistency
        calib_df = pd.DataFrame([calibration_results])
        calib_df.to_csv(calibration_path, index=False, float_format="%.6f")
        
        # Save calibrated metrics
        calibrated_metrics_path = get_metrics_calibrated_path(
            context.results_root, context.env_id, env_type, method, seed
        )
        ensure_dir_exists(calibrated_metrics_path, is_file=True)
        calibrated_metrics.to_csv(calibrated_metrics_path, index=False, float_format="%.6f")
        
        # Generate episode-level summary for consistency with Stage 4
        episode_summary = _generate_episode_summary(calibrated_metrics, context, env_type, method, seed, logger)
        
        logger.info(f"Calibration parameters saved: {calibration_path}")
        logger.info(f"Calibrated metrics saved: {calibrated_metrics_path}")
        logger.info(f"Calibration: temperature={calibration_results['temperature']:.3f}, "
                   f"ECE: {calibration_results['ece_before']:.3f} → {calibration_results['ece_after']:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _standard_temperature_scaling(q_means: np.ndarray, q_stds: np.ndarray, 
                                true_returns: np.ndarray, method: str, 
                                logger: logging.Logger) -> Dict[str, float]:
    """
    Standard Temperature Scaling following Guo et al. 2017.
    
    Minimizes Expected Calibration Error (ECE) - the industry standard.
    No bias correction - preserves true prediction quality for research.
    """
    # Ensure minimal uncertainty for deterministic methods
    if method.lower() in ['dqn'] and np.all(q_stds < 1e-5):
        q_stds = np.full_like(q_stds, 1e-6)
        logger.info(f"Using minimal uncertainty for {method} calibration")
    
    def compute_ece_and_reliability(temperature: float, n_bins: int = 10) -> Tuple[float, Dict]:
        """
        Compute Expected Calibration Error and reliability diagram data.
        
        Standard implementation following Guo et al. 2017.
        """
        # Scale uncertainties by temperature  
        scaled_stds = q_stds / temperature
        
        # Convert to confidence scores: higher uncertainty = lower confidence
        # Using normalized inverse relationship
        max_uncertainty = np.percentile(scaled_stds, 95)  # Robust normalization
        confidences = 1.0 - (scaled_stds / (max_uncertainty + 1e-8))
        confidences = np.clip(confidences, 0.0, 1.0)
        
        # Binary accuracy: within 1-sigma interval (68% for Gaussian)
        margins = scaled_stds  # 1-sigma
        accuracies = (np.abs(true_returns - q_means) <= margins).astype(float)
        
        # Bin by confidence for reliability diagram
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        reliability_data = {
            'bin_accuracies': [],
            'bin_confidences': [],
            'bin_counts': []
        }
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this confidence bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_count = np.sum(in_bin)
            
            if bin_count > 0:
                # Average confidence and accuracy in this bin
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                bin_weight = bin_count / len(confidences)
                
                # Add to ECE
                ece += bin_weight * abs(avg_confidence - avg_accuracy)
                
                # Store for reliability diagram
                reliability_data['bin_accuracies'].append(avg_accuracy)
                reliability_data['bin_confidences'].append(avg_confidence)
                reliability_data['bin_counts'].append(bin_count)
            else:
                reliability_data['bin_accuracies'].append(0.0)
                reliability_data['bin_confidences'].append((bin_lower + bin_upper) / 2)
                reliability_data['bin_counts'].append(0)
        
        return ece, reliability_data
    
    # Search for optimal temperature
    # Standard range used in literature
    temperature_candidates = np.linspace(0.1, 5.0, 50)  # Broader, finer search
    
    best_temperature = 1.0
    best_ece = float('inf')
    best_reliability = None
    
    for temp in temperature_candidates:
        try:
            ece, reliability_data = compute_ece_and_reliability(temp)
            if ece < best_ece:
                best_ece = ece
                best_temperature = temp
                best_reliability = reliability_data
        except:
            continue
    
    # Compute ECE before calibration (temperature = 1.0)
    ece_before, _ = compute_ece_and_reliability(1.0)
    
    logger.debug(f"Standard calibration for {method}: "
                f"temperature={best_temperature:.3f}, ECE: {ece_before:.3f} → {best_ece:.3f}")
    
    return {
        'temperature': best_temperature,
        'ece_before': ece_before,
        'ece_after': best_ece,
        'calibration_improvement': ece_before - best_ece,
        'reliability_data': best_reliability
    }


def _compute_standard_calibrated_metrics(filtered_dataset: pd.DataFrame,
                                       q_means: np.ndarray, q_stds: np.ndarray, 
                                       true_returns: np.ndarray,
                                       calibration_results: Dict[str, float],
                                       method: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Compute standard calibrated metrics focusing on research-relevant measures.
    """
    try:
        temperature = calibration_results['temperature']
        n_samples = len(filtered_dataset)
        
        # Apply temperature scaling
        calibrated_stds = q_stds / temperature
        
        # Compute standard research metrics
        
        # 1. Confidence scores (for reliability analysis)
        max_uncertainty = np.percentile(calibrated_stds, 95)
        confidences = 1.0 - (calibrated_stds / (max_uncertainty + 1e-8))
        confidences = np.clip(confidences, 0.0, 1.0)
        
        # 2. Binary accuracies (1-sigma coverage)
        margins = calibrated_stds
        accuracies = (np.abs(true_returns - q_means) <= margins).astype(float)
        
        # 3. Expected Calibration Error per sample
        # For analysis: how much each sample contributes to miscalibration
        ece_contributions = np.abs(confidences - accuracies)
        
        # 4. Standard prediction metrics
        prediction_errors = np.abs(true_returns - q_means)
        prediction_biases = true_returns - q_means
        normalized_errors = prediction_errors / np.maximum(calibrated_stds, 1e-6)
        
        # 5. Coverage at multiple levels (for interval prediction evaluation)
        coverage_levels = [0.5, 0.68, 0.8, 0.9, 0.95]  # Standard statistical levels
        coverage_results = {}
        
        for level in coverage_levels:
            z_score = stats.norm.ppf(0.5 + level/2)
            margins_level = z_score * calibrated_stds
            within_interval = (np.abs(true_returns - q_means) <= margins_level).astype(float)
            coverage_results[f'coverage_{int(level*100)}'] = within_interval
        
        # Compute traditional UQ metrics (consistent with Stage 4)
        # CRPS for Gaussian predictions
        def compute_crps_gaussian(y_true, mu, sigma):
            sigma = np.maximum(sigma, 1e-6)
            z = (y_true - mu) / sigma
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cdf_z = stats.norm.cdf(z)
                pdf_z = stats.norm.pdf(z)
            return sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1/np.sqrt(np.pi))
        
        crps_values = compute_crps_gaussian(true_returns, q_means, calibrated_stds)
        crps_values = np.nan_to_num(crps_values, nan=0.0, posinf=1000.0, neginf=0.0)
        
        # WIS (Weighted Interval Score) for 90% CI
        alpha = 0.1  # 90% CI
        z_score = stats.norm.ppf(1 - alpha/2)
        lower = q_means - z_score * calibrated_stds
        upper = q_means + z_score * calibrated_stds
        
        width = upper - lower
        undercoverage = np.maximum(0, lower - true_returns)
        overcoverage = np.maximum(0, true_returns - upper)
        coverage_penalty = (2/alpha) * (undercoverage + overcoverage)
        wis_values = width + coverage_penalty
        
        # ACE (Average Calibration Error) across multiple levels
        coverage_levels = [0.5, 0.8, 0.9]
        coverage_errors = []
        picp_results = {}
        
        for level in coverage_levels:
            z = stats.norm.ppf(0.5 + level/2)
            margin = z * calibrated_stds
            within_interval = np.abs(true_returns - q_means) <= margin
            actual_coverage = within_interval.astype(float)
            
            picp_results[f'picp_{int(level*100)}'] = actual_coverage
            coverage_errors.append(np.abs(actual_coverage - level))
        
        ace_values = np.mean(coverage_errors, axis=0)
        
        # Build results DataFrame combining traditional UQ + modern calibration metrics
        results_dict = {
            # Traditional UQ metrics (consistent with Stage 4)
            'crps': crps_values,
            'wis': wis_values,
            'ace': ace_values,
            **picp_results,
            'interval_width': width,
            
            # Modern calibration metrics  
            'confidence': confidences,
            'accuracy': accuracies,
            'ece': ece_contributions,
            
            # Additional prediction quality metrics
            'prediction_error': prediction_errors,
            'prediction_bias': prediction_biases,
            'normalized_error': normalized_errors,
            'calibrated_uncertainty': calibrated_stds,
            
            # Extended coverage analysis
            **coverage_results,
            
            # Metadata
            'state_id': np.arange(n_samples),
            'episode_id': filtered_dataset['episode_id'].values,
            'step': filtered_dataset['step_id'].values,
            'action_taken': filtered_dataset['action'].values,
            'seed': filtered_dataset['seed'].values,
            'algorithm': [method] * n_samples
        }
        
        calibrated_metrics = pd.DataFrame(results_dict)
        
        logger.info(f"Standard calibrated metrics computed: {len(calibrated_metrics):,} records")
        logger.info(f"Average ECE contribution: {np.mean(ece_contributions):.3f}")
        logger.info(f"68% coverage (1-sigma): {np.mean(coverage_results['coverage_68']):.3f}")
        
        return calibrated_metrics
        
    except Exception as e:
        logger.error(f"Failed to compute standard calibrated metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_supported_calibration_methods() -> list:
    """Get list of supported calibration methods."""
    return ['temperature_scaling']


def _generate_episode_summary(calibrated_metrics: pd.DataFrame, context: ExperimentContext,
                            env_type: str, method: str, seed: int, logger: logging.Logger) -> bool:
    """
    Generate episode-level summary consistent with Stage 4 format.
    
    This provides aggregated metrics at episode level for comparison with Stage 4 results.
    """
    try:
        # Group by episode and aggregate
        episode_groups = calibrated_metrics.groupby('episode_id')
        
        episode_data = []
        for episode_id, group in episode_groups:
            # Aggregate traditional UQ metrics
            episode_summary = {
                'episode_id': episode_id,
                'episode_length': len(group),
                'crps': group['crps'].mean(),
                'wis': group['wis'].mean(),
                'ace': group['ace'].mean(),
                'picp_50': group['picp_50'].mean(),
                'picp_90': group['picp_90'].mean(),
                'interval_width': group['interval_width'].mean(),
                
                # Additional calibration metrics
                'ece': group['ece'].mean(),
                'confidence': group['confidence'].mean(),
                'accuracy': group['accuracy'].mean(),
                'prediction_error': group['prediction_error'].mean(),
                'calibrated_uncertainty': group['calibrated_uncertainty'].mean(),
                
                # Coverage rates
                'coverage_68': group['coverage_68'].mean(),
                'coverage_90': group['coverage_90'].mean(),
                
                # Metadata
                'seed': seed,
                'algorithm': method
            }
            episode_data.append(episode_summary)
        
        # Save episode-level summary
        episode_df = pd.DataFrame(episode_data)
        
        # Use a different filename to avoid conflicts
        from ..utils.path_manager import get_result_dir
        result_dir = get_result_dir(context.results_root, context.env_id, env_type, method, seed)
        episode_summary_path = result_dir / "metrics_episode_calibrated.csv"
        ensure_dir_exists(episode_summary_path, is_file=True)
        
        episode_df.to_csv(episode_summary_path, index=False, float_format="%.6f")
        
        logger.info(f"Episode-level summary saved: {episode_summary_path}")
        logger.info(f"Episode summary: {len(episode_df)} episodes, "
                   f"avg CRPS={episode_df['crps'].mean():.3f}, "
                   f"avg ECE={episode_df['ece'].mean():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate episode summary: {e}")
        return False


def get_supported_calibration_methods() -> list:
    """Get list of supported calibration methods."""
    return ['temperature_scaling']


def validate_calibration_params(params: Dict[str, Any]) -> bool:
    """Validate calibration parameters structure."""
    required_keys = ['temperature', 'ece_before', 'ece_after']
    
    if not all(key in params for key in required_keys):
        return False
        
    # Check parameter value ranges
    if not (0.1 <= params['temperature'] <= 10.0):
        return False
    if not (0.0 <= params['ece_before'] <= 1.0):
        return False
    if not (0.0 <= params['ece_after'] <= 1.0):
        return False
        
    return True