"""
Stage 2: Performance Evaluation
Evaluate trained model performance on clean datasets.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from ..utils.context import ExperimentContext
from ..utils.logging_utils import get_stage_logger, StageTimer, log_stage_progress
from ..utils.path_manager import (
    get_performance_path, get_clean_dataset_path, 
    get_result_dir, ensure_dir_exists
)
from ..utils.data_format import (
    save_performance_metrics, load_performance_metrics,
    load_dataframe, verify_file_integrity
)


def run(context: ExperimentContext) -> bool:
    """
    Stage 2: Evaluate performance of trained models on clean datasets.
    
    This stage:
    1. Iterates through all (env_type, method, seed) combinations
    2. Loads evaluation datasets and computes performance metrics
    3. Saves performance results to standardized format
    4. Supports resumption (skips completed evaluations)
    
    Args:
        context: Experiment context with configuration
        
    Returns:
        True if all performance evaluations completed successfully
    """
    logger = get_stage_logger("stage2_performance")
    
    with StageTimer(logger, "Performance Evaluation") as timer:
        logger.info("=== Stage 2: Performance Evaluation ===")
        
        combinations = context.get_env_method_seed_combinations()
        total_combinations = len(combinations)
        
        success_count = 0
        
        # Group by (env_type, method) for efficient processing
        grouped_combinations = {}
        for env_type, method, seed in combinations:
            key = (env_type, method)
            if key not in grouped_combinations:
                grouped_combinations[key] = []
            grouped_combinations[key].append(seed)
        
        for (env_type, method), seeds in grouped_combinations.items():
            logger.info(f"Processing {method} on {env_type} with {len(seeds)} seeds")
            
            try:
                # Process all seeds for this env_type/method combination
                performance_results = _evaluate_algorithm_performance(
                    context, env_type, method, seeds, logger
                )
                
                if performance_results:
                    # Save aggregated performance results
                    # Create summary path (not per-seed)
                    output_path = Path(context.results_root) / context.env_id / env_type / method / "performance_summary.json"
                    ensure_dir_exists(output_path.parent)
                    
                    save_performance_metrics(performance_results, output_path)
                    
                    logger.info(f"Saved performance results for {method} on {env_type}: {output_path}")
                    success_count += len(seeds)
                else:
                    logger.error(f"Failed to evaluate {method} on {env_type}")
                    
            except Exception as e:
                logger.error(f"Error evaluating {method} on {env_type}: {e}")
        
        logger.info(f"Performance evaluation completed: {success_count}/{total_combinations} successful")
        return success_count == total_combinations


def _evaluate_algorithm_performance(context: ExperimentContext, env_type: str, 
                                   method: str, seeds: list, logger) -> Dict[str, Any]:
    """
    Evaluate performance for a single algorithm across multiple seeds
    
    Args:
        context: Experiment context
        env_type: Environment type (e.g., uncertainty_degradation_noise0.000)
        method: Algorithm name (e.g., qrdqn)
        seeds: List of seeds to evaluate
        logger: Logger instance
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Load evaluation dataset
        dataset_path = get_clean_dataset_path(Path(context.data_root), context.env_id, env_type, method)
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return None
        
        # Load dataset
        if dataset_path.suffix == '.xz':
            import pandas as pd
            df = pd.read_pickle(dataset_path, compression='xz')
        else:
            logger.error(f"Unsupported dataset format: {dataset_path}")
            return None
        
        # Filter for this algorithm (use 'algorithm' column from Stage 1)
        algo_data = df[df['algorithm'] == method]
        
        if len(algo_data) == 0:
            logger.error(f"No data found for algorithm {method} in dataset")
            return None
        
        # Compute performance metrics for each seed
        seed_results = []
        seed_rewards = []
        
        for seed in seeds:
            seed_data = algo_data[algo_data['seed'] == seed]
            
            if len(seed_data) == 0:
                logger.warning(f"No data found for seed {seed}")
                continue
            
            # Calculate episode returns by grouping by episode_id
            episode_returns = seed_data.groupby('episode_id')['reward'].sum()
            
            # Compute metrics for this seed
            mean_return = episode_returns.mean()
            std_return = episode_returns.std()
            n_episodes = len(episode_returns)
            
            seed_results.append({
                'seed': seed,
                'mean_return': mean_return,
                'std_return': std_return,
                'n_episodes': n_episodes
            })
            
            seed_rewards.append(mean_return)
        
        if not seed_rewards:
            logger.error(f"No valid seed data found for {method}")
            return None
        
        # Compute overall statistics
        overall_mean = np.mean(seed_rewards)
        overall_std = np.std(seed_rewards)
        
        # Determine status based on performance thresholds
        success_threshold = context.success_threshold
        
        if overall_mean >= success_threshold + 40:  # 240 for LunarLander
            status = "EXCELLENT"
        elif overall_mean >= success_threshold:  # 200 for LunarLander
            status = "SOLVED"
        elif overall_mean >= success_threshold - 50:  # 150 for LunarLander
            status = "BORDERLINE"
        else:
            status = "UNSOLVED"
        
        # Prepare results
        performance_results = {
            'algorithm': method,
            'env_type': env_type,
            'overall_stats': {
                'mean_return': overall_mean,
                'std_return': overall_std,
                'n_seeds': len(seed_rewards),
                'status': status,
                'success_threshold': success_threshold
            },
            'seed_results': seed_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"{method} performance: {overall_mean:.1f}±{overall_std:.1f} ({status})")
        
        return performance_results
        
    except Exception as e:
        logger.error(f"Error in performance evaluation: {e}")
        return None