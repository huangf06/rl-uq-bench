"""
Stage 3: Q-value Distribution Extractor
Extract Q-value distributions from evaluation datasets with quantile values.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from ..utils.context import ExperimentContext
from ..utils.logging_utils import get_stage_logger, StageTimer, log_stage_progress
from ..utils.path_manager import (
    get_q_values_path, get_clean_dataset_path,
    get_result_dir, ensure_dir_exists
)
from ..utils.data_format import (
    save_q_values, load_q_values, load_dataframe,
    verify_file_integrity, get_file_size_mb
)


def run(context: ExperimentContext) -> bool:
    """
    Stage 3: Extract Q-value distributions from evaluation datasets.
    
    This stage:
    1. Iterates through all (env_type, method) combinations
    2. Loads evaluation datasets with quantile values from Stage 1
    3. Extracts and organizes Q-value distributions per seed
    4. Saves Q-value distributions to compressed format
    5. Supports resumption (skips completed extractions)
    
    Args:
        context: Experiment context with configuration
        
    Returns:
        True if all Q-value extractions completed successfully
    """
    logger = get_stage_logger("stage3_q_extractor")
    
    with StageTimer(logger, "Q-value Extraction") as timer:
        logger.info("=== Stage 3: Q-value Distribution Extractor ===")
        
        # Group by (env_type, method) for efficient processing
        env_method_combinations = [(env_type, method) for env_type in context.env_types for method in context.algorithms]
        total_combinations = len(env_method_combinations)
        
        success_count = 0
        
        for i, (env_type, method) in enumerate(env_method_combinations, 1):
            log_stage_progress(logger, "Q-value Extraction", i, total_combinations, "combinations")
            
            # Run Q-value extraction for this env_type/method combination
            success = _extract_q_values_for_env_method(context, env_type, method, logger)
            
            if success:
                success_count += 1
            else:
                logger.warning(f"Failed Q-value extraction: {env_type}/{method}")
        
        logger.info(f"Q-value extraction completed: {success_count}/{total_combinations} successful")
        return success_count == total_combinations


def _extract_q_values_for_env_method(context: ExperimentContext, env_type: str, 
                                   method: str, logger: logging.Logger) -> bool:
    """
    Extract Q-values for all seeds of a specific (env_type, method) combination.
    
    Args:
        context: Experiment context
        env_type: Environment type identifier
        method: UQ method name
        logger: Logger instance
        
    Returns:
        True if Q-value extraction succeeded for all seeds
    """
    logger.info(f"Extracting Q-values: {env_type}/{method}")
    
    try:
        # Load evaluation dataset from Stage 1
        dataset_path = get_clean_dataset_path(Path(context.data_root), context.env_id, env_type, method)
        
        if not dataset_path.exists():
            logger.error(f"Evaluation dataset not found: {dataset_path}")
            return False
        
        # Load dataset
        logger.info(f"Loading dataset: {dataset_path}")
        df = pd.read_pickle(dataset_path, compression='xz')
        logger.info(f"Loaded {len(df)} samples from dataset")
        
        # Process each seed separately
        success_count = 0
        total_seeds = len(context.seeds)
        
        for seed in context.seeds:
            success = _extract_q_values_for_seed(context, env_type, method, seed, df, logger)
            if success:
                success_count += 1
        
        logger.info(f"Q-value extraction for {env_type}/{method}: {success_count}/{total_seeds} seeds successful")
        return success_count == total_seeds
        
    except Exception as e:
        logger.error(f"Q-value extraction failed for {env_type}/{method}: {e}")
        return False


def _extract_q_values_for_seed(context: ExperimentContext, env_type: str, method: str, 
                              seed: int, df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Extract Q-values for a specific seed from the evaluation dataset.
    
    Args:
        context: Experiment context
        env_type: Environment type identifier
        method: UQ method name
        seed: Random seed
        df: Full evaluation dataset
        logger: Logger instance
        
    Returns:
        True if Q-value extraction succeeded
    """
    try:
        # Check if already extracted
        q_values_path = get_q_values_path(
            Path(context.results_root), context.env_id, env_type, method, seed
        )
        
        if _q_values_exist_and_valid(q_values_path, logger):
            logger.debug(f"Q-values already exist for seed {seed}")
            return True
        
        # Filter dataset for this seed
        seed_data = df[df['seed'] == seed].copy()
        if len(seed_data) == 0:
            logger.warning(f"No data found for seed {seed}")
            return False
        
        # Extract Q-value distributions based on method type
        if method == "qrdqn":
            q_values_data = _extract_qrdqn_q_values(seed_data, logger)
        elif method == "qr_bootstrap_dqn":
            q_values_data = _extract_qr_bootstrap_q_values(seed_data, logger)
        elif method == "bootstrapped_dqn":
            q_values_data = _extract_bootstrapped_q_values(seed_data, logger)
        elif method == "mcdropout_dqn":
            q_values_data = _extract_mcdropout_q_values(seed_data, logger)
        elif method == "dqn":
            q_values_data = _extract_dqn_q_values(seed_data, logger)
        else:
            logger.warning(f"Unsupported method for Q-value extraction: {method}")
            return False
        
        if q_values_data is None:
            logger.error(f"Failed to extract Q-values for seed {seed}")
            return False
        
        # Prepare metadata
        quantile_shape = None
        if 'quantile_values' in q_values_data.columns and len(q_values_data) > 0:
            quantile_shape = q_values_data['quantile_values'].iloc[0].shape
        
        metadata = {
            'method': method,
            'env_type': env_type,
            'seed': seed,
            'num_samples': len(q_values_data),
            'extraction_time': pd.Timestamp.now().isoformat(),
            'quantiles_shape': str(quantile_shape) if quantile_shape is not None else 'None'
        }
        
        # Save Q-values
        q_values_path.parent.mkdir(parents=True, exist_ok=True)
        _save_q_values_with_metadata(q_values_data, q_values_path, metadata)
        
        file_size = get_file_size_mb(q_values_path)
        logger.info(f"Q-values saved for seed {seed}: {q_values_path} ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"Q-value extraction failed for seed {seed}: {e}")
        return False


def _q_values_exist_and_valid(q_values_path: Path, logger: logging.Logger) -> bool:
    """
    Check if Q-values already exist and are valid.
    
    Args:
        q_values_path: Path to q_values.xz file
        logger: Logger instance
        
    Returns:
        True if Q-values file exists and passes validation
    """
    if not q_values_path.exists():
        return False
    
    try:
        # Basic file integrity check
        if q_values_path.stat().st_size < 100:  # Too small to be valid
            return False
        
        # Try to load a small sample to verify format
        test_df = pd.read_pickle(q_values_path, compression='xz')
        if len(test_df) == 0:
            return False
        
        # Check for required columns
        required_cols = ['q_mean', 'q_std', 'episode_id', 'step_id']
        if not all(col in test_df.columns for col in required_cols):
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Q-values validation failed: {e}")
        return False


def _extract_qrdqn_q_values(seed_data: pd.DataFrame, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Extract Q-values from QRDQN quantile data in the evaluation dataset.
    
    Args:
        seed_data: Dataset filtered for specific seed
        logger: Logger instance
        
    Returns:
        DataFrame with Q-value distributions or None if extraction failed
    """
    try:
        # Extract quantile columns (quantile_0, quantile_1, ..., quantile_169)
        quantile_cols = [col for col in seed_data.columns if col.startswith('quantile_')]
        if not quantile_cols:
            logger.error("No quantile columns found in dataset")
            return None
        
        logger.debug(f"Found {len(quantile_cols)} quantile columns")
        
        # Extract quantile values as numpy arrays
        quantile_values = []
        for _, row in seed_data.iterrows():
            quantiles = np.array([row[col] for col in quantile_cols])
            quantile_values.append(quantiles)
        
        # Prepare Q-values data
        q_values_data = seed_data[['episode_id', 'step_id', 'action', 'reward', 'done', 'q_mean', 'q_std']].copy()
        q_values_data['quantile_values'] = quantile_values
        q_values_data['num_quantiles'] = len(quantile_cols)
        
        # Add distributional statistics
        q_values_data['q_min'] = [np.min(qv) for qv in quantile_values]
        q_values_data['q_max'] = [np.max(qv) for qv in quantile_values] 
        q_values_data['q_median'] = [np.median(qv) for qv in quantile_values]
        q_values_data['q_iqr'] = [np.percentile(qv, 75) - np.percentile(qv, 25) for qv in quantile_values]
        
        logger.info(f"Extracted Q-values for {len(q_values_data)} samples with {len(quantile_cols)} quantiles each")
        return q_values_data
        
    except Exception as e:
        logger.error(f"Failed to extract QRDQN Q-values: {e}")
        return None


def _save_q_values_with_metadata(q_values_data: pd.DataFrame, path: Path, metadata: Dict[str, Any]):
    """
    Save Q-values data with metadata to compressed format.
    
    Args:
        q_values_data: Q-values DataFrame
        path: Output file path
        metadata: Metadata dictionary
    """
    # Add metadata as attributes to the DataFrame
    for key, value in metadata.items():
        q_values_data.attrs[key] = value
    
    # Save to compressed pickle
    q_values_data.to_pickle(path, compression='xz')


def _extract_bootstrapped_q_values(seed_data: pd.DataFrame, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Extract Q-values from Bootstrapped DQN ensemble data in the evaluation dataset.
    
    Args:
        seed_data: Dataset filtered for specific seed
        logger: Logger instance
        
    Returns:
        DataFrame with Q-value distributions or None if extraction failed
    """
    try:
        # Bootstrapped DQN data format: quantile_0, quantile_1, ..., quantile_98 columns
        # Extract quantile columns (same logic as QRDQN)
        quantile_cols = [col for col in seed_data.columns if col.startswith('quantile_')]
        if not quantile_cols:
            logger.error("No quantile columns found for Bootstrapped DQN")
            return None
        
        logger.debug(f"Found {len(quantile_cols)} quantile columns for BootstrappedDQN")
        
        # Extract quantile values as numpy arrays
        quantile_values = []
        for _, row in seed_data.iterrows():
            quantiles = np.array([row[col] for col in quantile_cols])
            quantile_values.append(quantiles)
        
        # Prepare Q-values data (same format as QRDQN for compatibility)
        q_values_data = seed_data[['episode_id', 'step_id', 'action', 'reward', 'done', 'q_mean', 'q_std']].copy()
        q_values_data['quantile_values'] = quantile_values
        q_values_data['num_quantiles'] = len(quantile_cols)
        
        # Add distributional statistics
        q_values_data['q_min'] = [np.min(qv) for qv in quantile_values]
        q_values_data['q_max'] = [np.max(qv) for qv in quantile_values] 
        q_values_data['q_median'] = [np.median(qv) for qv in quantile_values]
        q_values_data['q_iqr'] = [np.percentile(qv, 75) - np.percentile(qv, 25) for qv in quantile_values]
        
        logger.info(f"Extracted BootstrappedDQN Q-values for {len(q_values_data)} samples with {len(quantile_cols)} quantiles each")
        return q_values_data
        
    except Exception as e:
        logger.error(f"Failed to extract Bootstrapped DQN Q-values: {e}")
        return None


def _extract_mcdropout_q_values(seed_data: pd.DataFrame, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Extract Q-values from MC Dropout sampling data in the evaluation dataset.
    
    Args:
        seed_data: Dataset filtered for specific seed
        logger: Logger instance
        
    Returns:
        DataFrame with Q-value distributions or None if extraction failed
    """
    try:
        # MC Dropout should have samples stored in quantile_values from Stage1
        if 'quantile_values' not in seed_data.columns:
            logger.error("No quantile_values column found for MC Dropout DQN")
            return None
        
        # Prepare Q-values data (same format as QRDQN for compatibility)
        q_values_data = seed_data[['episode_id', 'step_id', 'action', 'reward', 'done', 'q_mean', 'q_std']].copy()
        q_values_data['quantile_values'] = seed_data['quantile_values']
        
        # Determine number of quantiles from first non-null entry
        first_quantiles = None
        for val in q_values_data['quantile_values']:
            if val is not None:
                first_quantiles = np.array(val)
                break
        
        num_quantiles = len(first_quantiles) if first_quantiles is not None else 99
        q_values_data['num_quantiles'] = num_quantiles
        
        logger.info(f"Extracted MC Dropout Q-values with {num_quantiles} quantiles")
        return q_values_data
        
    except Exception as e:
        logger.error(f"Failed to extract MC Dropout Q-values: {e}")
        return None


def _extract_dqn_q_values(seed_data: pd.DataFrame, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Extract Q-values from standard DQN with constructed pseudo-uncertainty.
    
    Args:
        seed_data: Dataset filtered for specific seed
        logger: Logger instance
        
    Returns:
        DataFrame with Q-value distributions or None if extraction failed
    """
    try:
        # DQN should have basic Q-values and constructed quantiles from Stage1
        if 'quantile_values' not in seed_data.columns:
            logger.error("No quantile_values column found for DQN")
            return None
        
        # Prepare Q-values data (same format as QRDQN for compatibility)
        q_values_data = seed_data[['episode_id', 'step_id', 'action', 'reward', 'done', 'q_mean', 'q_std']].copy()
        q_values_data['quantile_values'] = seed_data['quantile_values']
        
        # DQN uses constructed Gaussian quantiles (99 quantiles from Stage1)
        num_quantiles = 99
        q_values_data['num_quantiles'] = num_quantiles
        
        logger.info(f"Extracted DQN Q-values with constructed {num_quantiles} quantiles")
        return q_values_data
        
    except Exception as e:
        logger.error(f"Failed to extract DQN Q-values: {e}")
        return None


def _extract_qr_bootstrap_q_values(seed_data: pd.DataFrame, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Extract Q-values from QR-Bootstrap fusion method data in the evaluation dataset.
    
    QR-Bootstrap combines quantile regression (QRDQN) with bootstrap ensembles,
    so it should have both quantile values and bootstrap ensemble data.
    
    Args:
        seed_data: Dataset filtered for specific seed
        logger: Logger instance
        
    Returns:
        DataFrame with Q-value distributions or None if extraction failed
    """
    try:
        # QR-Bootstrap should have quantile columns from QR component
        quantile_cols = [col for col in seed_data.columns if col.startswith('quantile_')]
        if not quantile_cols:
            logger.error("No quantile columns found for QR-Bootstrap DQN")
            return None
        
        logger.debug(f"Found {len(quantile_cols)} quantile columns for QR-Bootstrap")
        
        # Extract quantile values as numpy arrays
        quantile_values = []
        for _, row in seed_data.iterrows():
            quantiles = np.array([row[col] for col in quantile_cols])
            quantile_values.append(quantiles)
        
        # Prepare Q-values data (similar format to QRDQN but with fusion info)
        q_values_data = seed_data[['episode_id', 'step_id', 'action', 'reward', 'done', 'q_mean', 'q_std']].copy()
        q_values_data['quantile_values'] = quantile_values
        q_values_data['num_quantiles'] = len(quantile_cols)
        
        # Add distributional statistics from quantiles
        q_values_data['q_min'] = [np.min(qv) for qv in quantile_values]
        q_values_data['q_max'] = [np.max(qv) for qv in quantile_values] 
        q_values_data['q_median'] = [np.median(qv) for qv in quantile_values]
        q_values_data['q_iqr'] = [np.percentile(qv, 75) - np.percentile(qv, 25) for qv in quantile_values]
        
        # Add fusion-specific metadata
        q_values_data['method_type'] = 'qr_bootstrap_fusion'
        
        # The q_std from Stage 1 should already incorporate both QR and Bootstrap uncertainties
        logger.info(f"Extracted QR-Bootstrap Q-values for {len(q_values_data)} samples with {len(quantile_cols)} quantiles each")
        return q_values_data
        
    except Exception as e:
        logger.error(f"Failed to extract QR-Bootstrap Q-values: {e}")
        return None


# Additional utility functions can be added here as needed