"""
Stage 4b: Fine-Grained UQ Analysis (Trigger-Based)

Implements detailed state-action level analysis when needed:
- Triggered when method differences < 0.5σ or manual request
- Random 5% sampling of state-action pairs
- Full distributional analysis for deep-dive investigation
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import warnings
from ..utils.context import ExperimentContext
from ..utils.logging_utils import get_stage_logger, StageTimer
from ..utils.path_manager import (
    get_clean_dataset_path, ensure_dir_exists
)
from ..utils.data_format import load_dataframe


def should_trigger_fine_analysis(episode_metrics_results: Dict[str, pd.DataFrame], 
                                threshold_sigma: float = 0.5) -> bool:
    """
    Determine if fine-grained analysis should be triggered.
    
    Triggers when:
    1. Method differences in core metrics < threshold_sigma
    2. High variance in episode-level results
    3. Manual request for detailed analysis
    
    Args:
        episode_metrics_results: Results from Stage 4a by method
        threshold_sigma: Threshold for triggering (default 0.5σ)
        
    Returns:
        True if fine-grained analysis is needed
    """
    if len(episode_metrics_results) < 2:
        return False  # Need at least 2 methods to compare
        
    # Compare methods on core metrics
    methods = list(episode_metrics_results.keys())
    core_metrics = ['crps', 'wis', 'ace']
    
    for metric in core_metrics:
        method_means = []
        method_stds = []
        
        for method in methods:
            df = episode_metrics_results[method]
            if metric in df.columns:
                method_means.append(df[metric].mean())
                method_stds.append(df[metric].std())
        
        if len(method_means) >= 2:
            # Calculate effect size between methods
            diff = abs(method_means[0] - method_means[1])
            pooled_std = np.sqrt((method_stds[0]**2 + method_stds[1]**2) / 2)
            
            if pooled_std > 0:
                effect_size = diff / pooled_std
                if effect_size < threshold_sigma:
                    return True  # Small effect size - need fine-grained analysis
    
    return False


def run_fine_grained_analysis(context: ExperimentContext, 
                            sample_ratio: float = 0.05,
                            focus_high_uncertainty: bool = True) -> bool:
    """
    Run fine-grained UQ analysis on sampled state-action pairs.
    
    Args:
        context: Experiment context
        sample_ratio: Fraction of state-action pairs to sample (default 5%)
        focus_high_uncertainty: If True, sample high-uncertainty states preferentially
        
    Returns:
        True if analysis completed successfully
    """
    logger = get_stage_logger("stage4b_finegrained")
    
    with StageTimer(logger, "Fine-Grained UQ Analysis") as timer:
        logger.info("=== Stage 4b: Fine-Grained UQ Analysis ===")
        logger.info(f"Sampling {sample_ratio*100:.1f}% of state-action pairs")
        
        combinations = context.get_env_method_seed_combinations()
        success_count = 0
        
        for env_type, method, seed in combinations:
            logger.info(f"Fine-grained analysis: {env_type}/{method}/seed_{seed}")
            
            success = _run_single_fine_analysis(
                context, env_type, method, seed, sample_ratio, 
                focus_high_uncertainty, logger
            )
            
            if success:
                success_count += 1
        
        logger.info(f"Fine-grained analysis completed: {success_count}/{len(combinations)} successful")
        return success_count == len(combinations)


def _run_single_fine_analysis(context: ExperimentContext, env_type: str, method: str, 
                            seed: int, sample_ratio: float, focus_high_uncertainty: bool,
                            logger: logging.Logger) -> bool:
    """Run fine-grained analysis for single combination."""
    try:
        # Load evaluation dataset
        dataset_path = get_clean_dataset_path(context.data_root, context.env_id, env_type, method)
        
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return False
            
        dataset = load_dataframe(dataset_path)
        filtered_dataset = dataset[
            (dataset['algorithm'] == method) & 
            (dataset['seed'] == seed)
        ]
        
        if len(filtered_dataset) == 0:
            logger.error(f"No data found for {method}/seed_{seed}")
            return False
        
        # Sample state-action pairs
        sampled_data = _sample_state_actions(
            filtered_dataset, sample_ratio, focus_high_uncertainty, logger
        )
        
        if len(sampled_data) == 0:
            logger.warning("No data after sampling")
            return False
            
        # Compute detailed UQ analysis
        fine_analysis_results = _compute_fine_grained_metrics(
            sampled_data, method, logger
        )
        
        # Save results
        output_path = _get_fine_analysis_path(
            context.results_root, context.env_id, env_type, method, seed
        )
        ensure_dir_exists(output_path, is_file=True)
        
        fine_analysis_results.to_csv(output_path, index=False, float_format="%.6f")
        logger.info(f"Fine analysis saved: {output_path} ({len(fine_analysis_results)} records)")
        
        return True
        
    except Exception as e:
        logger.error(f"Fine-grained analysis failed: {e}")
        return False


def _sample_state_actions(dataset: pd.DataFrame, sample_ratio: float, 
                        focus_high_uncertainty: bool, logger: logging.Logger) -> pd.DataFrame:
    """
    Sample state-action pairs for detailed analysis.
    
    Two strategies:
    1. Random sampling: Uniform random selection
    2. High-uncertainty sampling: Focus on high-variance states
    """
    n_total = len(dataset)
    n_sample = max(1, int(n_total * sample_ratio))
    
    if focus_high_uncertainty:
        # Strategy B: Sample high-uncertainty states
        quantile_columns = [col for col in dataset.columns if col.startswith('quantile_')]
        
        if len(quantile_columns) > 0:
            # Compute Q-value variance for each state-action
            quantile_matrix = dataset[quantile_columns].values
            q_variances = np.var(quantile_matrix, axis=1)
            
            # Sample preferentially from high-variance states
            # Top 20% get 80% of samples, rest get 20%
            variance_threshold = np.percentile(q_variances, 80)
            high_var_mask = q_variances >= variance_threshold
            
            n_high_var = int(n_sample * 0.8)
            n_low_var = n_sample - n_high_var
            
            high_var_data = dataset[high_var_mask]
            low_var_data = dataset[~high_var_mask]
            
            sampled_high = high_var_data.sample(
                n=min(len(high_var_data), n_high_var), 
                random_state=42
            )
            sampled_low = low_var_data.sample(
                n=min(len(low_var_data), n_low_var),
                random_state=42
            )
            
            sampled_data = pd.concat([sampled_high, sampled_low], ignore_index=True)
            logger.info(f"High-uncertainty sampling: {len(sampled_high)} high-var + {len(sampled_low)} low-var")
            
        else:
            # Fallback to random sampling
            sampled_data = dataset.sample(n=n_sample, random_state=42)
            logger.info(f"Fallback to random sampling: {len(sampled_data)} records")
    else:
        # Strategy A: Random sampling
        sampled_data = dataset.sample(n=n_sample, random_state=42)
        logger.info(f"Random sampling: {len(sampled_data)} records")
    
    return sampled_data


def _compute_fine_grained_metrics(sampled_data: pd.DataFrame, method: str,
                                logger: logging.Logger) -> pd.DataFrame:
    """
    Compute detailed UQ metrics for sampled state-action pairs.
    
    Includes:
    - Core 6 metrics from Stage 4a
    - Distributional analysis (skewness, kurtosis, etc.)
    - Risk measures (CVaR)
    - Policy analysis metrics
    """
    quantile_columns = [col for col in sampled_data.columns if col.startswith('quantile_')]
    
    if len(quantile_columns) == 0:
        logger.error("No quantile columns found")
        return pd.DataFrame()
    
    if 'remaining_return' not in sampled_data.columns:
        logger.error("Missing remaining_return column")
        return pd.DataFrame()
    
    logger.info(f"Computing fine-grained metrics for {len(sampled_data)} samples")
    
    # Extract data
    quantiles_matrix = sampled_data[quantile_columns].values
    remaining_returns = sampled_data['remaining_return'].values
    
    q_means = np.mean(quantiles_matrix, axis=1)
    q_stds = np.maximum(np.std(quantiles_matrix, axis=1), 1e-6)
    
    # Core metrics (same as Stage 4a)
    crps_values = _compute_crps_vectorized(remaining_returns, q_means, q_stds)
    wis_values = _compute_wis_vectorized(remaining_returns, q_means, q_stds)
    picp_50 = _compute_picp_vectorized(remaining_returns, q_means, q_stds, 0.5)
    picp_90 = _compute_picp_vectorized(remaining_returns, q_means, q_stds, 0.9)
    
    # Fine-grained distributional metrics
    distributional_metrics = _compute_distributional_features(quantiles_matrix)
    
    # Risk measures for QRDQN
    risk_metrics = _compute_risk_measures(quantiles_matrix, method)
    
    # Policy analysis
    policy_metrics = _compute_policy_analysis(quantiles_matrix)
    
    # Build results DataFrame
    results = {
        # Core metrics
        'crps': crps_values,
        'wis': wis_values,
        'picp_50': picp_50,
        'picp_90': picp_90,
        'prediction_error': np.abs(remaining_returns - q_means),
        'prediction_bias': remaining_returns - q_means,
        
        # Distributional features
        **distributional_metrics,
        
        # Risk measures
        **risk_metrics,
        
        # Policy analysis
        **policy_metrics,
        
        # Metadata
        'state_id': range(len(sampled_data)),
        'episode_id': sampled_data['episode_id'].values,
        'step': sampled_data['step_id'].values,
        'action': sampled_data['action'].values,
        'seed': sampled_data['seed'].values,
        'algorithm': [method] * len(sampled_data)
    }
    
    return pd.DataFrame(results)


def _compute_distributional_features(quantiles_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute detailed distributional features."""
    features = {}
    
    # Basic statistics
    features['q_min'] = np.min(quantiles_matrix, axis=1)
    features['q_max'] = np.max(quantiles_matrix, axis=1)
    features['q_median'] = np.median(quantiles_matrix, axis=1)
    features['q_iqr'] = np.percentile(quantiles_matrix, 75, axis=1) - np.percentile(quantiles_matrix, 25, axis=1)
    
    # Higher-order moments
    features['q_skewness'] = np.array([stats.skew(row) if len(row) > 2 else 0.0 for row in quantiles_matrix])
    features['q_kurtosis'] = np.array([stats.kurtosis(row) if len(row) > 3 else 0.0 for row in quantiles_matrix])
    
    return features


def _compute_risk_measures(quantiles_matrix: np.ndarray, method: str) -> Dict[str, np.ndarray]:
    """Compute risk-sensitive measures."""
    risk_metrics = {}
    
    if method.lower() == 'qrdqn':
        # CVaR at 5% and 10% levels
        n_quantiles = quantiles_matrix.shape[1]
        if n_quantiles >= 10:
            n_5pct = max(1, int(n_quantiles * 0.05))
            n_10pct = max(1, int(n_quantiles * 0.10))
            
            risk_metrics['cvar_5'] = np.mean(quantiles_matrix[:, :n_5pct], axis=1)
            risk_metrics['cvar_10'] = np.mean(quantiles_matrix[:, :n_10pct], axis=1)
        else:
            risk_metrics['cvar_5'] = np.min(quantiles_matrix, axis=1)
            risk_metrics['cvar_10'] = np.min(quantiles_matrix, axis=1)
    
    return risk_metrics


def _compute_policy_analysis(quantiles_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute policy-related uncertainty analysis."""
    # For single-action quantiles, compute uncertainty characteristics
    q_means = np.mean(quantiles_matrix, axis=1)
    q_stds = np.std(quantiles_matrix, axis=1)
    
    policy_metrics = {
        'uncertainty_magnitude': q_stds,
        'confidence_score': 1.0 / (1.0 + q_stds),  # Higher for lower uncertainty
        'value_optimism': q_means - np.median(quantiles_matrix, axis=1),  # Mean vs median difference
    }
    
    return policy_metrics


# Reuse vectorized computation functions from Stage 4a
def _compute_crps_vectorized(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Vectorized CRPS computation."""
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


def _compute_picp_vectorized(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, confidence: float) -> np.ndarray:
    """Vectorized PICP computation."""
    z = stats.norm.ppf(0.5 + confidence/2)
    margin = z * sigma
    within_interval = np.abs(y_true - mu) <= margin
    return within_interval.astype(float)


def _get_fine_analysis_path(results_root: Path, env_id: str, env_type: str, 
                          algorithm: str, seed: int) -> Path:
    """Generate path for fine-grained analysis results."""
    return results_root / env_id / env_type / algorithm / f"seed_{seed}" / "metrics_finegrained.csv"