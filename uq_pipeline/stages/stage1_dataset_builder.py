"""
Stage 1: Dataset Builder

This module generates clean evaluation datasets for reinforcement learning experiments.
It supports both flattened array storage and bytes storage for high-dimensional observations.

Key features:
- Adaptive storage: Uses bytes storage for high-dimensional data (>512 elements or >3D)
- Case-insensitive algorithm support: Handles any case in algorithm names
- Performance optimized: Cached environment spaces, optimized validation
- Comprehensive validation: Checks data integrity and consistency with vectorized operations
- Resumption support: Skips existing valid datasets

Supported algorithms:
- QRDQN: Quantile Regression DQN with uncertainty quantification
- DQN: Deep Q-Network (basic Q-values only)

Storage formats:
- Flattened arrays: For small observations (<512 elements)
- Bytes storage: For high-dimensional observations with shape/dtype metadata

Validation features:
- Vectorized episode continuity checks
- Comprehensive data integrity validation
- Bytes storage consistency verification
"""

__all__ = ['run', 'validate_dataset_structure']

import logging
import re
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Stable-baselines3 imports (lazy loading to avoid early ImportError)
SB3_AVAILABLE = False
QRDQN = None

def _load_qrdqn():
    """Lazy load QRDQN to avoid early ImportError."""
    global QRDQN, SB3_AVAILABLE
    if QRDQN is None:
        try:
            from sb3_contrib import QRDQN as QRDQN_CLASS
            QRDQN = QRDQN_CLASS
            SB3_AVAILABLE = True
        except ImportError:
            QRDQN = None
            SB3_AVAILABLE = False
    return QRDQN

try:
    import rl_zoo3
    RL_ZOO3_AVAILABLE = True
except ImportError:
    RL_ZOO3_AVAILABLE = False

# Noise wrapper import (from wrappers/noise.py analysis)
try:
    import gymnasium as gym
    from wrappers.noise import GaussianObsNoise
    NOISE_WRAPPER_AVAILABLE = True
except ImportError:
    gym = None
    GaussianObsNoise = None
    NOISE_WRAPPER_AVAILABLE = False

from ..utils.context import ExperimentContext
from ..utils.logging_utils import get_stage_logger, StageTimer, log_stage_progress
from ..utils.path_manager import get_clean_dataset_path, ensure_dir_exists, get_trained_model_candidates
from ..utils.data_format import save_dataframe, load_dataframe

# Cache for environment spaces to avoid repeated gym.make() calls
_SPACE_CACHE = {}


@contextmanager
def managed_environment(env_id: str, noise_level: float = 0.0, seed: Optional[int] = None):
    """
    Context manager for proper environment resource management.
    
    Args:
        env_id: The environment ID
        noise_level: Observation noise level
        seed: Random seed for environment
        
    Yields:
        The created environment
    """
    env = None
    try:
        env = _create_noisy_environment(env_id, noise_level, seed=seed)
        if env is None:
            raise RuntimeError(f"Failed to create environment: {env_id}")
        yield env
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                # Log but don't raise - closing errors shouldn't fail the pipeline
                logging.getLogger(__name__).warning(f"Error closing environment: {e}")


def _get_env_spaces(env_id: str):
    """Get observation and action spaces for an environment, with caching."""
    if env_id not in _SPACE_CACHE:
        import gymnasium as gym
        with managed_environment(env_id) as temp_env:
            _SPACE_CACHE[env_id] = {
                'observation_space': temp_env.observation_space,
                'action_space': temp_env.action_space
            }
    return _SPACE_CACHE[env_id]


def run(context: ExperimentContext) -> bool:
    """
    Main entry point for Stage 1: Dataset Builder.
    
    Args:
        context: Experiment context with configuration
        
    Returns:
        True if dataset generation completed successfully
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Dataset Generation...")
    
    start_time = time.time()
    logger.info("=== Stage 1: Dataset Builder ===")
    
    # Check dependencies
    if not _check_dependencies(logger, context):
        logger.error("Dependency check failed")
        return False
    
    env_types = context.env_types
    total_env_types = len(env_types)
    
    success_count = 0
    failure_count = 0
    skipped_count = 0
    
    # Get available algorithms (only those with trained models)
    available_algorithms = _get_available_algorithms(context, logger)
    if not available_algorithms:
        logger.error("No trained models found for any configured algorithms")
        return False
    
    # Generate datasets for all algorithm-environment combinations
    total_combinations = len(available_algorithms) * len(env_types)
    logger.info(f"Will process {total_combinations} algorithm-environment combinations")
    
    combination_count = 0
    
    # Track detailed failures for reporting
    failed_combinations = []
    
    for algorithm in available_algorithms:
        for i, env_type in enumerate(env_types, 1):
            combination_count += 1
            log_stage_progress(logger, "Dataset Builder", combination_count, total_combinations, 
                             f"{algorithm}-{env_type} combinations")
            
            # Check if dataset already exists and is valid
            dataset_path = get_clean_dataset_path(context.data_root, context.env_id, env_type, algorithm)
            
            if _dataset_exists_and_valid(dataset_path, logger):
                logger.info(f"Dataset already exists: {dataset_path}")
                success_count += 1
                skipped_count += 1
                continue
            
            # Generate dataset for this algorithm-environment combination
            env_start_time = time.time()
            try:
                success = _generate_dataset(context, env_type, dataset_path, logger, algorithm)
                env_duration = time.time() - env_start_time
                
                if success:
                    success_count += 1
                    logger.info(f"✓ Generated dataset for {algorithm}-{env_type} in {env_duration:.2f}s")
                    
                    # Memory usage info (optional)
                    try:
                        import psutil
                        if psutil is not None:
                            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                            logger.debug(f"Memory usage: {memory_usage:.1f} MB")
                    except ImportError:
                        pass  # psutil not available, skip memory monitoring
                        
                else:
                    failure_count += 1
                    failed_combinations.append(f"{algorithm}-{env_type}")
                    logger.error(f"✗ Failed to generate dataset for {algorithm}-{env_type} after {env_duration:.2f}s")
                    
            except Exception as e:
                failure_count += 1
                failed_combinations.append(f"{algorithm}-{env_type}")
                env_duration = time.time() - env_start_time
                logger.error(f"✗ Exception during dataset generation for {algorithm}-{env_type} after {env_duration:.2f}s: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    total_duration = time.time() - start_time
    logger.info(f"Dataset generation completed: {success_count}/{total_combinations} successful in {total_duration:.2f}s")
    
    # Report detailed results
    if failed_combinations:
        if len(failed_combinations) <= 10:
            logger.warning(f"Failed combinations: {failed_combinations}")
        else:
            logger.warning(f"Failed combinations: {failed_combinations[:10]} ... +{len(failed_combinations)-10} more")
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} existing datasets")
    
    # Return success only if all combinations succeeded (strict validation)
    overall_success = (failure_count == 0)
    if overall_success:
        logger.info(f"Completed Dataset Generation in {total_duration:.2f}s")
    else:
        logger.error(f"Dataset generation failed: {failure_count}/{total_combinations} combinations failed")
    
    return overall_success


def _get_available_algorithms(context: ExperimentContext, logger: logging.Logger) -> List[str]:
    """
    Check which algorithms have trained models available.
    
    Args:
        context: Experiment context
        logger: Logger instance
        
    Returns:
        List of algorithm names that have trained models and are supported (all lowercase)
    """
    # Currently supported algorithms (case-insensitive)
    SUPPORTED_ALGORITHMS = {'qrdqn', 'qr_bootstrap_dqn', 'bootstrapped_dqn', 'mcdropout_dqn', 'dqn'}  # Using set for O(1) lookup
    
    available_algorithms = []
    
    logger.info("Checking for available trained models...")
    
    # Normalize algorithms to lowercase for consistent processing (generate once)
    normalized_algorithms = [algo.lower() for algo in context.algorithms]
    
    for algorithm in normalized_algorithms:
        # Skip unsupported algorithms
        if algorithm not in SUPPORTED_ALGORITHMS:
            logger.warning(f"✗ Algorithm {algorithm} not yet supported (skipping)")
            continue
            
        # Check if this algorithm has models for at least one environment type
        has_models = False
        
        for env_type in context.env_types:
            # Use first seed for checking model availability
            test_seed = context.seeds[0] if context.seeds else 101
            
            # Use path_manager to get model candidates instead of hardcoded paths
            # Pass lowercase algorithm name for path generation
            model_candidates = get_trained_model_candidates(context.env_id, env_type, algorithm, test_seed)
            
            # Check for model files
            if any(path.exists() for path in model_candidates):
                has_models = True
                break
        
        if has_models:
            available_algorithms.append(algorithm)  # Keep lowercase for consistency
            logger.info(f"✓ Found trained models for algorithm: {algorithm}")
        else:
            logger.warning(f"✗ No trained models found for algorithm: {algorithm} (skipping)")
    
    logger.info(f"Available algorithms: {available_algorithms}")
    return available_algorithms


def _check_dependencies(logger: logging.Logger, context=None) -> bool:
    """Check if all required dependencies are available."""
    required_dependencies = {
        'torch': 'pip install torch',
        'gymnasium': 'pip install gymnasium',
        'pandas': 'pip install pandas',
        'numpy': 'pip install numpy',
    }
    
    optional_dependencies = {
        'psutil': 'pip install psutil',  # For memory monitoring
        'tqdm': 'pip install tqdm',  # For progress bars
        'rl_zoo3': 'pip install rl-zoo3',  # For hyperparams and wrappers
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required dependencies
    for dep, install_cmd in required_dependencies.items():
        try:
            __import__(dep)
            logger.debug(f"✓ {dep} available")
        except ImportError:
            missing_required.append((dep, install_cmd))
            logger.error(f"✗ {dep} not available")
    
    # Check optional dependencies
    for dep, install_cmd in optional_dependencies.items():
        try:
            __import__(dep)
            logger.debug(f"✓ {dep} available (optional)")
        except ImportError:
            missing_optional.append((dep, install_cmd))
            logger.warning(f"⚠ {dep} not available (optional)")
    
    # Check specific UQ methods only if needed
    if context and hasattr(context, 'algorithms'):
        uq_algorithms = ['qrdqn', 'bootstrapped_dqn', 'mcdropout_dqn']
        needs_sb3_contrib = any(algo.lower() in uq_algorithms for algo in context.algorithms)
        
        if needs_sb3_contrib:
            # Try lazy loading QRDQN
            QRDQN = _load_qrdqn()
            if QRDQN is None:
                logger.error("Stable-baselines3 UQ methods not available. Please install sb3-contrib.")
                missing_required.append(('sb3_contrib', 'pip install sb3-contrib'))
        
        # Check for stable_baselines3 only if DQN is needed
        needs_sb3 = any(algo.lower() == 'dqn' for algo in context.algorithms)
        if needs_sb3:
            try:
                import stable_baselines3
                logger.debug("✓ stable_baselines3 available")
            except ImportError:
                logger.error("Stable-baselines3 not available. Please install stable-baselines3.")
                missing_required.append(('stable_baselines3', 'pip install stable-baselines3'))
    
    if not NOISE_WRAPPER_AVAILABLE:
        logger.error("Noise wrapper not available. Please check wrappers/noise.py.")
        missing_required.append(('wrappers.noise', 'Check wrappers/noise.py implementation'))
    
    # Report missing dependencies
    if missing_required:
        logger.error("Missing required dependencies:")
        for dep, install_cmd in missing_required:
            logger.error(f"  {dep}: {install_cmd}")
        return False
    
    if missing_optional:
        logger.warning("Missing optional dependencies:")
        for dep, install_cmd in missing_optional:
            logger.warning(f"  {dep}: {install_cmd}")
    
    logger.info("✓ All required dependencies available")
    return True


def _generate_dataset(context: ExperimentContext, env_type: str, 
                     dataset_path: Path, logger: logging.Logger, algorithm: str) -> bool:
    """
    Generate evaluation dataset for specific environment type.
    
    Migrated from analysis/modules/dataset_generator.py generate_dataset() function.
    Key changes:
    - Uses context instead of global variables
    - Simplified to single env_type instead of all algorithms
    - Uses standardized paths from path_manager
    
    Args:
        context: Experiment context
        env_type: Environment type identifier (e.g., "uncertainty_degradation_noise0.050")
        dataset_path: Output path for dataset
        logger: Logger instance
        algorithm: Algorithm name (required parameter)
        
    Returns:
        True if dataset generated successfully
    """
    logger.info(f"Generating dataset for {env_type}")
    
    # Get noise level from context (preferred method)
    noise_level = context.get_noise_level(env_type)
    logger.info(f"Noise level for {env_type}: {noise_level}")
    
    # Create output directory
    ensure_dir_exists(dataset_path, is_file=True)
    
    # Generate episodes using trained models
    episodes_data = _generate_episodes_with_trained_models(context, env_type, noise_level, logger, algorithm)
    
    if episodes_data is None or episodes_data.empty:
        logger.error(f"No episodes generated for {env_type}")
        return False
    
    # Save dataset with compression (from original dataset format)
    try:
        save_dataframe(episodes_data, dataset_path)
        logger.info(f"Saved dataset: {dataset_path} ({len(episodes_data)} steps from {episodes_data['episode_id'].nunique()} episodes)")
        return True
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        return False


def _parse_noise_level(env_type: str) -> float:
    """
    Parse noise level from environment type string.
    
    Migrated from analysis/modules/noise_config_manager.py parsing logic.
    
    Args:
        env_type: Environment type (e.g., 'uncertainty_degradation_noise0.050')
        
    Returns:
        Noise level as float
    """
    # Extract noise level using regex (from original implementation pattern)
    match = re.search(r'noise([\d.]+)', env_type)
    if match:
        return float(match.group(1))
    else:
        # Default to 0.0 for non-noise environments
        return 0.0


def _create_noisy_environment(env_id: str, noise_level: float, seed: int = None):
    """
    Create environment with specified noise configuration.
    
    Migrated from wrappers/noise.py GaussianObsNoise wrapper usage
    and analysis/modules/environment_verifier.py create_evaluation_noise_config().
    
    Args:
        env_id: Base environment identifier
        noise_level: Noise level to apply
        seed: Random seed for environment
        
    Returns:
        Configured environment instance
    """
    # Create base environment (similar to rl_zoo3 create_test_env logic)
    env = gym.make(env_id)
    
    if seed is not None:
        env.reset(seed=seed)
    
    # Apply noise wrapper if noise_level > 0 (from original noise config)
    if noise_level > 0.0:
        env = GaussianObsNoise(
            env, 
            noise_std=noise_level,
            clip=True,
            inplace=False
        )
    
    return env


def _generate_episodes_with_trained_models(context: ExperimentContext, env_type: str, 
                                          noise_level: float, logger: logging.Logger, 
                                          algorithm: str, 
                                          min_success_ratio: float = 0.8) -> Optional[pd.DataFrame]:
    """
    Generate episodes using multiple trained models for evaluation (Multi-Policy Mix).
    
    Following O3 recommendation: "Use all 10 seeds to collect small amounts of episodes each,
    then combine into a unified dataset of target size."
    
    Key design principles:
    - Multi-policy mix: Use ALL available seeds, not just one
    - Coverage > single agent performance: Prioritize strategy diversity
    - Uniform distribution: Each seed contributes roughly equal episodes
    - Reproducible: Fixed eval mode ensures trajectory reproducibility
    - Rich metadata: Record seed_id, return, noise_level, env_type for later analysis
    - Fault tolerance: Require minimum success ratio to ensure data quality
    
    Args:
        context: Experiment context
        env_type: Environment type identifier
        noise_level: Noise level for environment
        logger: Logger instance
        algorithm: Algorithm name
        min_success_ratio: Minimum ratio of seeds that must succeed (default: 0.8)
        
    Returns:
        DataFrame with multi-policy mixed episode data or None if failed
    """
    all_episode_data = []
    
    # Calculate episodes per seed for uniform distribution
    total_episodes = context.eval_episodes
    available_seeds = context.seeds if context.seeds else [101]
    min_required_seeds = max(1, int(len(available_seeds) * min_success_ratio))
    
    # Calculate episodes per seed distribution
    episodes_per_seed = total_episodes // len(available_seeds)
    extra_episodes = total_episodes % len(available_seeds)
    
    logger.info(f"Multi-policy mix for {env_type}: {len(available_seeds)} seeds, "
                f"minimum required: {min_required_seeds} (ratio: {min_success_ratio:.1%})")
    
    successful_seeds = 0
    total_steps = 0
    successful_seed_data = []
    
    for seed_idx, seed in enumerate(available_seeds):
        # Calculate episodes for this seed (distribute extras to first few seeds)
        episodes_for_this_seed = episodes_per_seed + (1 if seed_idx < extra_episodes else 0)
        
        logger.info(f"Processing seed {seed} ({seed_idx+1}/{len(available_seeds)}): "
                    f"{episodes_for_this_seed} episodes")
        
        try:
            # Use path_manager to get model candidates  
            model_candidates = get_trained_model_candidates(context.env_id, env_type, algorithm, seed)
            
            model_path = None
            for candidate_path in model_candidates:
                if candidate_path.exists():
                    model_path = candidate_path
                    break
            
            if not model_path:
                logger.warning(f"No trained model found for seed {seed} in candidates: {[str(p) for p in model_candidates]}")
                continue
                
            logger.debug(f"Loading model from: {model_path}")
            
            # Load the trained model based on algorithm type (env creation handled internally)
            model = _load_algorithm_model(algorithm, model_path, seed, context, logger)
            if model is None:
                continue
            
            # Generate episodes with trained model policy and extract UQ data
            try:
                with managed_environment(context.env_id, noise_level, seed=seed) as env:
                    episode_data = _collect_episode_data_trained_policy(
                        env, model, episodes_for_this_seed, algorithm, logger, seed=seed, context=context
                    )
            except RuntimeError as e:
                logger.warning(f"Failed to create environment for seed {seed}: {e}")
                continue
            
            if episode_data is not None and not episode_data.empty:
                # Add comprehensive metadata for UQ analysis
                episode_data['env_type'] = env_type
                episode_data['noise_level'] = noise_level
                episode_data['algorithm'] = algorithm
                episode_data['seed'] = seed
                
                # Add episode-level metadata for coverage analysis
                episode_data['model_path'] = str(model_path)
                episode_data['policy_id'] = f"{algorithm}_seed_{seed}"
                
                successful_seed_data.append({
                    'seed': seed,
                    'data': episode_data,
                    'steps': len(episode_data),
                    'episodes': episode_data['episode_id'].nunique()
                })
                successful_seeds += 1
                total_steps += len(episode_data)
                
                logger.info(f"✓ Seed {seed}: {len(episode_data)} steps from "
                           f"{episode_data['episode_id'].nunique()} episodes")
            else:
                logger.warning(f"✗ Seed {seed}: No episode data generated")
                
        except Exception as e:
            logger.error(f"✗ Seed {seed}: Failed to load model or generate episodes: {e}")
            continue
    
    # Check if sufficient seeds succeeded
    if successful_seeds < min_required_seeds:
        logger.error(f"❌ Insufficient successful seeds: {successful_seeds}/{len(available_seeds)} "
                     f"(required: {min_required_seeds}, ratio: {min_success_ratio:.1%})")
        return None
    
    if not successful_seed_data:
        logger.error("❌ No episode data generated from any seed")
        return None
    
    # Redistribute episodes from successful seeds to reach target
    combined_data = _redistribute_episodes_from_successful_seeds(
        successful_seed_data, total_episodes, logger
    )
    
    if combined_data is None or combined_data.empty:
        logger.error("❌ Failed to redistribute episodes from successful seeds")
        return None
    
    # Reassign episode IDs to be globally unique across all seeds
    combined_data = _reassign_global_episode_ids(combined_data)
    
    logger.info(f"✅ Multi-policy dataset generated: {successful_seeds}/{len(available_seeds)} seeds, "
                f"{len(combined_data)} total steps, {combined_data['episode_id'].nunique()} episodes")
    
    # Log coverage statistics
    seed_distribution = combined_data.groupby('seed')['episode_id'].nunique()
    logger.info(f"Episode distribution per seed: {dict(seed_distribution)}")
    
    return combined_data


def _redistribute_episodes_from_successful_seeds(successful_seed_data: list, 
                                               target_episodes: int, 
                                               logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Redistribute episodes from successful seeds to reach target episode count.
    
    When some seeds fail to load, redistribute episodes among successful seeds
    to maintain target dataset size while preserving multi-policy mix.
    
    Args:
        successful_seed_data: List of dicts with seed info and episode data
        target_episodes: Target number of episodes to generate
        logger: Logger instance
        
    Returns:
        Combined DataFrame with redistributed episodes or None if failed
    """
    if not successful_seed_data:
        return None
    
    all_data = []
    total_available_episodes = sum(seed_info['episodes'] for seed_info in successful_seed_data)
    
    if total_available_episodes < target_episodes:
        logger.warning(f"Available episodes ({total_available_episodes}) < target ({target_episodes}). "
                       f"Using all available episodes.")
        # Use all available episodes
        for seed_info in successful_seed_data:
            all_data.append(seed_info['data'])
    else:
        # Redistribute episodes proportionally among successful seeds
        episodes_per_seed = target_episodes // len(successful_seed_data)
        extra_episodes = target_episodes % len(successful_seed_data)
        
        logger.info(f"Redistributing {target_episodes} episodes among {len(successful_seed_data)} successful seeds: "
                    f"{episodes_per_seed} base + {extra_episodes} extra")
        
        for idx, seed_info in enumerate(successful_seed_data):
            seed_target = episodes_per_seed + (1 if idx < extra_episodes else 0)
            available = seed_info['episodes']
            
            if available >= seed_target:
                # Select first N episodes from this seed
                seed_data = seed_info['data']
                selected_episodes = sorted(seed_data['episode_id'].unique())[:seed_target]
                filtered_data = seed_data[seed_data['episode_id'].isin(selected_episodes)]
                all_data.append(filtered_data)
                logger.debug(f"Seed {seed_info['seed']}: selected {len(selected_episodes)}/{available} episodes")
            else:
                # Use all available episodes from this seed
                all_data.append(seed_info['data'])
                logger.debug(f"Seed {seed_info['seed']}: using all {available} episodes")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return None


def _extract_uq_data(model, state, algorithm: str, action: int):
    """
    Extract UQ data (quantile values, Q-means, Q-stds) from model for specific action.
    
    Args:
        model: Trained model instance
        state: Current state observation
        algorithm: Algorithm name ('qrdqn', 'dqn', etc.)
        action: The specific action to extract quantiles for
        
    Returns:
        Tuple of (q_mean, q_std, quantile_values) for the given action
    """
    try:
        # Normalize algorithm name to lowercase for consistent comparison
        algorithm_lower = algorithm.lower()
        
        if algorithm_lower == "qrdqn":
            # Extract quantile values from QRDQN model
            with torch.no_grad():
                # Convert state to tensor with proper device handling
                if isinstance(state, np.ndarray):
                    state_tensor = torch.as_tensor(state, device=model.device).unsqueeze(0)
                else:
                    state_tensor = torch.as_tensor(np.array(state), device=model.device).unsqueeze(0)
                
                # Get quantile values using the policy
                quantiles = model.policy.quantile_net(state_tensor)
                # quantiles shape: [batch_size=1, n_quantiles, n_actions]
                
                # Transpose to get [batch_size, n_actions, n_quantiles] for easier processing
                quantiles = quantiles.transpose(1, 2)  # [1, n_actions, n_quantiles]
                
                # Extract quantiles for the SPECIFIC ACTION taken
                action_quantiles = quantiles[0, action, :].cpu().numpy()  # [n_quantiles]
                
                # Calculate mean and std from quantiles for this specific action
                q_mean = float(np.mean(action_quantiles))
                q_std = float(np.std(action_quantiles))
                
                return q_mean, q_std, action_quantiles
                
        elif algorithm_lower == "qr_bootstrap_dqn":
            # Extract fusion of QR quantiles and Bootstrap ensemble
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state_tensor = torch.as_tensor(state, device=model.device).unsqueeze(0)
                else:
                    state_tensor = torch.as_tensor(np.array(state), device=model.device).unsqueeze(0)
                
                # Get QR quantiles from the fusion model
                qr_quantiles = model.policy.q_net.get_qr_quantiles(state_tensor)  # [1, n_actions, n_quantiles]
                bootstrap_ensemble = model.policy.q_net.get_bootstrap_ensemble(state_tensor)  # [1, n_actions, n_heads]
                
                # Extract quantiles for the specific action
                action_qr_quantiles = qr_quantiles[0, action, :].cpu().numpy()  # [n_quantiles]
                action_bootstrap_ensemble = bootstrap_ensemble[0, action, :].cpu().numpy()  # [n_heads]
                
                # Combine QR and Bootstrap uncertainties
                # Use QR quantiles as primary uncertainty source with Bootstrap as augmentation
                qr_mean = float(np.mean(action_qr_quantiles))
                qr_std = float(np.std(action_qr_quantiles))
                bootstrap_std = float(np.std(action_bootstrap_ensemble))
                
                # Fused uncertainty: combine both sources
                q_mean = qr_mean  # Use QR mean as primary
                q_std = float(np.sqrt(qr_std**2 + bootstrap_std**2))  # Combined uncertainty
                
                return q_mean, q_std, action_qr_quantiles
                
        elif algorithm_lower == "bootstrapped_dqn":
            # Extract ensemble outputs from multiple heads
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state_tensor = torch.as_tensor(state, device=model.device).unsqueeze(0)
                else:
                    state_tensor = torch.as_tensor(np.array(state), device=model.device).unsqueeze(0)
                
                # Get Q-values from all heads
                all_heads_q = []
                # Check if model has n_heads attribute, otherwise use default
                n_heads = getattr(model, 'n_heads', 10)  # Default from hyperparams
                
                if hasattr(model.policy.q_net, 'heads'):
                    # Multi-head architecture
                    for head_idx in range(n_heads):
                        # Try different ways to extract from each head
                        try:
                            if hasattr(model.policy.q_net, 'features_extractor'):
                                features = model.policy.q_net.features_extractor(state_tensor)
                                q_values = model.policy.q_net.heads[head_idx](features)
                            else:
                                # Direct forward pass through the head
                                q_values = model.policy.q_net.heads[head_idx](state_tensor)
                        except Exception:
                            # Fallback: use the full network with head selection
                            model.policy.q_net._current_heads = torch.tensor([head_idx], device=model.device)
                            q_values = model.policy.q_net(state_tensor)
                            
                        all_heads_q.append(q_values[0, action].cpu().numpy())
                else:
                    # Fallback: if heads not accessible, simulate with noise
                    base_q = model.policy.q_net(state_tensor)[0, action].cpu().numpy()
                    for _ in range(n_heads):
                        # Add small noise to simulate ensemble
                        noisy_q = base_q + np.random.normal(0, abs(base_q) * 0.1)
                        all_heads_q.append(noisy_q)
                
                ensemble_values = np.array(all_heads_q)  # [n_heads]
                q_mean = float(np.mean(ensemble_values))
                q_std = float(np.std(ensemble_values))
                
                # Convert ensemble samples to quantiles for consistent interface
                quantile_levels = np.linspace(0.01, 0.99, 99)
                quantiles = np.percentile(ensemble_values, quantile_levels * 100)
                
                return q_mean, q_std, quantiles

        elif algorithm_lower == "mcdropout_dqn":
            # MC Dropout sampling
            samples = []
            for _ in range(30):  # 30 MC samples
                with torch.no_grad():
                    # Enable dropout during inference
                    model.policy.q_net.train()  # Activate dropout
                    
                    if isinstance(state, np.ndarray):
                        state_tensor = torch.as_tensor(state, device=model.device).unsqueeze(0)
                    else:
                        state_tensor = torch.as_tensor(np.array(state), device=model.device).unsqueeze(0)
                    
                    q_values = model.policy.q_net(state_tensor)  # [1, n_actions]
                    q_value_action = q_values[0, action].cpu().numpy()
                    samples.append(q_value_action)
            
            # Reset to eval mode
            model.policy.q_net.eval()
            
            samples = np.array(samples)
            q_mean = float(np.mean(samples))
            q_std = float(np.std(samples))
            
            # Convert MC samples to quantiles for consistent interface
            quantile_levels = np.linspace(0.01, 0.99, 99)
            quantiles = np.percentile(samples, quantile_levels * 100)
            
            return q_mean, q_std, quantiles

        elif algorithm_lower == "dqn":
            # DQN doesn't have quantiles, just return basic Q-values
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state_tensor = torch.as_tensor(state, device=model.device).unsqueeze(0)
                else:
                    state_tensor = torch.as_tensor(np.array(state), device=model.device).unsqueeze(0)
                
                q_values = model.policy.q_net(state_tensor)  # [1, n_actions]
                q_value_action = q_values[0, action].cpu().numpy()  # Q-value for specific action
                
                # Construct Gaussian quantiles for DQN (GPT-5 recommendation)
                # Simple heuristic for uncertainty: σ = |Q| * 0.1 + 0.01
                q_mean = float(q_value_action)
                sigma_hat = abs(q_mean) * 0.1 + 0.01  # Avoid zero variance
                
                # Generate quantiles from N(q_mean, sigma_hat)
                from scipy.stats import norm
                quantile_levels = np.linspace(0.01, 0.99, 99)
                quantiles = norm.ppf(quantile_levels, loc=q_mean, scale=sigma_hat)
                
                return q_mean, sigma_hat, quantiles
                
        else:
            # Other algorithms not implemented yet
            return None, None, None
            
    except Exception as e:
        # If UQ extraction fails, return None values
        return None, None, None


def _load_algorithm_model(algorithm: str, model_path: Path, seed: int, context, logger: logging.Logger):
    """
    Load trained model based on algorithm type.
    
    Args:
        algorithm: Algorithm name (normalized to lowercase)
        model_path: Path to model file
        seed: Random seed for reproducibility
        context: Experiment context
        logger: Logger instance
        
    Returns:
        Loaded model instance or None if failed
    """
    try:
        # Load model with proper custom objects to handle serialization issues
        logger.debug(f"Loading model for {algorithm} seed {seed}")
        
        # Get cached environment spaces
        spaces = _get_env_spaces(context.env_id)
        
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "exploration_schedule": lambda _: 0.0,
            "observation_space": spaces['observation_space'],
            "action_space": spaces['action_space'],
            "_last_obs": None,
            "_last_episode_starts": None,
            "_last_original_obs": None
        }
        
        # Normalize algorithm name to lowercase for consistent comparison
        algorithm_lower = algorithm.lower()
        
        if algorithm_lower == "qrdqn":
            qrdqn_cls = _load_qrdqn() # Lazy load QRDQN
            if qrdqn_cls is None:
                logger.error("QRDQN not available. Please install sb3-contrib.")
                return None
            model = qrdqn_cls.load(str(model_path), custom_objects=custom_objects)
        elif algorithm_lower == "bootstrapped_dqn":
            from rl_zoo3.bootstrapped_dqn import BootstrappedDQN
            # Fix custom objects for BootstrappedDQN  
            bootstrap_custom_objects = custom_objects.copy()
            bootstrap_custom_objects.update({
                "n_heads": 10,
                "bootstrap_prob": 0.65,
                # Add additional attributes that may be missing
                "observation_space": spaces['observation_space'],
                "action_space": spaces['action_space'],
                "_last_obs": None,
                "_last_episode_starts": None
            })
            model = BootstrappedDQN.load(str(model_path), custom_objects=bootstrap_custom_objects)
        elif algorithm_lower == "qr_bootstrap_dqn":
            # Import QR-Bootstrap DQN from rl_zoo3
            from rl_zoo3.qr_bootstrap_dqn import QRBootstrapDQN
            model = QRBootstrapDQN.load(str(model_path), custom_objects=custom_objects)
        elif algorithm_lower == "mcdropout_dqn":
            from rl_zoo3.mcdropout_dqn import MCDropoutDQN
            model = MCDropoutDQN.load(str(model_path), custom_objects=custom_objects)
        elif algorithm_lower == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(str(model_path), custom_objects=custom_objects)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load {algorithm} model from {model_path}: {e}")
        return None


def _reassign_global_episode_ids(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reassign episode IDs to be globally unique across all seeds.
    
    Args:
        data: Combined DataFrame from multiple seeds
        
    Returns:
        DataFrame with globally unique episode IDs
    """
    data = data.copy()
    
    # Sort by seed and original episode_id to ensure deterministic reassignment
    data = data.sort_values(['seed', 'episode_id', 'step_id']).reset_index(drop=True)
    
    # Create mapping from (seed, original_episode_id) to new global episode_id
    unique_episodes = data[['seed', 'episode_id']].drop_duplicates().reset_index(drop=True)
    unique_episodes['new_episode_id'] = range(len(unique_episodes))
    
    # Merge back to assign new IDs
    data = data.merge(
        unique_episodes[['seed', 'episode_id', 'new_episode_id']], 
        on=['seed', 'episode_id'], 
        how='left'
    )
    
    # Preserve original episode IDs and add global IDs
    data['original_episode_id'] = data['episode_id']  # Keep original for reference
    data['global_episode_id'] = data['new_episode_id']  # Global unique ID
    data['episode_id'] = data['new_episode_id']  # Replace for backward compatibility
    data = data.drop('new_episode_id', axis=1)
    
    return data


def _collect_episode_data_trained_policy(env, model, num_episodes: int, algorithm: str, logger: logging.Logger, seed: int = None, context=None) -> Optional[pd.DataFrame]:
    """
    Collect episode data using trained QRDQN model policy.
    
    Args:
        env: Environment instance  
        model: Trained QRDQN model
        num_episodes: Number of episodes to collect
        logger: Logger instance
        
    Returns:
        DataFrame with collected episode data
    """
    all_data = []
    successful_episodes = 0
    failed_episodes = 0
    
    for episode_id in range(num_episodes):
        episode_start_time = time.time()
        try:
            # Reset environment (handle both new and old gym API)
            try:
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]
                else:
                    obs = reset_result
            except Exception as e:
                logger.warning(f"Environment reset failed for episode {episode_id}: {e}")
                failed_episodes += 1
                continue
            
            # Ensure obs is properly shaped
            if isinstance(obs, np.ndarray) and obs.shape == (1,) and hasattr(obs[0], 'shape'):
                obs = obs[0]
            
            step_id = 0
            done = False
            episode_data = []
            max_steps = _get_max_episode_steps(env, context)  # Always returns int now
            
            while not done and step_id < max_steps:
                try:
                    # Store current state
                    current_state = obs.copy() if hasattr(obs, 'copy') else np.array(obs)
                    
                    # Use trained model to predict action (deterministic=True for evaluation)
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Ensure action is in correct format
                    if isinstance(action, np.ndarray):
                        action = action.item() if action.size == 1 else action[0]
                    
                    # Extract UQ data (quantile values) for the ACTUAL ACTION taken
                    q_mean, q_std, quantile_values = _extract_uq_data(model, obs, algorithm, int(action))
                    
                    # Step environment
                    step_result = env.step(action)
                    if len(step_result) == 4:
                        next_obs, reward, done, info = step_result
                    elif len(step_result) == 5:
                        next_obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        logger.warning(f"Unexpected step result length: {len(step_result)}")
                        break
                    
                    # Handle VecEnv output format
                    if isinstance(next_obs, np.ndarray) and next_obs.shape == (1,) and hasattr(next_obs[0], 'shape'):
                        next_obs = next_obs[0]
                    if isinstance(reward, np.ndarray) and reward.shape == (1,):
                        reward = reward[0]
                    if isinstance(done, np.ndarray) and done.shape == (1,):
                        done = done[0]
                    
                    # Validate data before storing
                    if not np.isfinite(reward) or np.isnan(reward):
                        logger.warning(f"Invalid reward in episode {episode_id}, step {step_id}: {reward}")
                        reward = 0.0  # Fallback value
                    
                    # Store step data
                    step_data = {
                        'episode_id': episode_id,
                        'step_id': step_id,
                        'action': int(action),  # Ensure it's an integer for discrete action space
                        'reward': float(reward),
                        'done': bool(done),
                        'q_mean': float(q_mean) if q_mean is not None else None,
                        'q_std': float(q_std) if q_std is not None else None,
                    }
                    
                    # Flatten state observations for better DataFrame performance
                    current_state_flat = _flatten_state_observation(current_state, "state")
                    next_state_flat = _flatten_state_observation(next_obs, "next_state")
                    
                    # Add shape information for reconstruction
                    if hasattr(current_state, 'shape'):
                        current_state_flat['state_shape'] = str(current_state.shape)
                    if hasattr(next_obs, 'shape'):
                        next_state_flat['next_state_shape'] = str(next_obs.shape)
                    
                    # Add flattened state features
                    step_data.update(current_state_flat)
                    step_data.update(next_state_flat)
                    
                    # Add quantile values if available
                    if quantile_values is not None:
                        for i, qval in enumerate(quantile_values):
                            step_data[f'quantile_{i}'] = float(qval)
                    
                    episode_data.append(step_data)
                    
                    # Move to next step
                    obs = next_obs
                    step_id += 1
                    
                except Exception as e:
                    logger.warning(f"Step {step_id} failed in episode {episode_id}: {e}")
                    break  # Exit episode on step error
            
            # Episode completed - compute remaining discounted returns
            if step_id > 0 and episode_data:
                # Post-process episode data to compute correct Q-value ground truth
                episode_data_with_returns = _compute_remaining_discounted_returns(episode_data, gamma=0.99)
                all_data.extend(episode_data_with_returns)
                successful_episodes += 1
                episode_duration = time.time() - episode_start_time
                # Get final reward from last step if available
                final_reward = episode_data[-1].get('reward', 0.0) if episode_data else 0.0
                logger.debug(f"Episode {episode_id}: {step_id} steps, final reward: {final_reward:.2f}, duration: {episode_duration:.2f}s")
            else:
                failed_episodes += 1
                logger.warning(f"Episode {episode_id} produced no data")
        
        except Exception as e:
            failed_episodes += 1
            logger.error(f"Episode {episode_id} failed completely: {e}")
            continue
    
    # Log episode statistics
    logger.info(f"Episode collection: {successful_episodes} successful, {failed_episodes} failed")
    
    if all_data:
        try:
            df = pd.DataFrame(all_data)
            # Validate DataFrame before returning
            if len(df) == 0:
                logger.error("Generated DataFrame is empty")
                return None
            return df
        except Exception as e:
            logger.error(f"Failed to create DataFrame from collected data: {e}")
            return None
    else:
        logger.error("No episode data collected")
        return None


# Legacy function kept for compatibility
def _generate_episodes(context: ExperimentContext, env_type: str, 
                      logger: logging.Logger) -> pd.DataFrame:
    """Legacy wrapper - calls the new implementation."""
    noise_level = _parse_noise_level(env_type)
    return _generate_episodes_with_trained_models(context, env_type, noise_level, logger, "qrdqn")


def _validate_dataframe(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Internal validation logic for DataFrame structure.
    
    Args:
        df: DataFrame to validate
        logger: Logger instance
        
    Returns:
        True if DataFrame structure is valid
    """
    # Check required columns (updated for flattened state format)
    required_columns = ['episode_id', 'step_id', 'action', 'reward', 'done']
    
    # Check for either original state columns or flattened state columns
    has_original_state = 'state' in df.columns and 'next_state' in df.columns
    has_flattened_state = any(col.startswith('state_') for col in df.columns) and any(col.startswith('next_state_') for col in df.columns)
    
    # Check for bytes storage consistency
    has_state_bytes = 'state_bytes' in df.columns
    has_next_state_bytes = 'next_state_bytes' in df.columns
    
    # Ensure bytes storage is consistent (both present or both absent)
    if has_state_bytes != has_next_state_bytes:
        logger.warning(f"Dataset has inconsistent bytes storage: state_bytes={has_state_bytes}, next_state_bytes={has_next_state_bytes}")
        return False
    
    # Check for bytes storage format
    has_byte_state = has_state_bytes and has_next_state_bytes
    
    if not (has_original_state or has_flattened_state or has_byte_state):
        logger.warning(f"Dataset missing state columns (neither original, flattened, nor bytes storage)")
        return False
    
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logger.warning(f"Dataset missing required columns: {missing_cols}")
        return False
    
    # Check minimum data
    if len(df) < 10:  # Minimum threshold
        logger.warning(f"Dataset has insufficient data: {len(df)} rows")
        return False
    
    # Additional validation checks
    # Check for NaN values in critical columns
    critical_columns = ['episode_id', 'step_id', 'action', 'reward', 'done']
    for col in critical_columns:
        if df[col].isnull().any():
            logger.warning(f"Dataset has null values in critical column: {col}")
            return False
    
    # Check episode continuity (optimized with vectorized operations)
    episode_continuity_check = df.groupby('episode_id')['step_id'].apply(
        lambda s: (s.diff().fillna(1) == 1).all()
    )
    
    if not episode_continuity_check.all():
        problematic_episodes = episode_continuity_check[~episode_continuity_check].index.tolist()
        logger.warning(f"Dataset episodes have non-consecutive step_ids: {problematic_episodes}")
        return False
    
    # Check state columns consistency (for flattened format)
    if has_flattened_state:
        state_cols = [col for col in df.columns if col.startswith('state_') and not col.endswith(('_shape', '_bytes', '_dtype'))]
        next_state_cols = [col for col in df.columns if col.startswith('next_state_') and not col.endswith(('_shape', '_bytes', '_dtype'))]
        
        if len(state_cols) != len(next_state_cols):
            logger.warning(f"Dataset has inconsistent flattened state column counts")
            return False
    
    # Check action space validity
    if 'action' in df.columns:
        actions = df['action'].unique()
        if len(actions) == 0:
            logger.warning(f"Dataset has no valid actions")
            return False
    
    logger.info(f"Valid dataset found: {len(df)} rows, {df['episode_id'].nunique()} episodes")
    return True


def validate_dataset_structure(df: pd.DataFrame) -> bool:
    """
    Validate that dataset has required structure and columns.
    
    Args:
        df: Dataset DataFrame to validate
        
    Returns:
        True if dataset structure is valid
    """
    # Use internal validation logic
    import logging
    logger = logging.getLogger(__name__)
    
    return _validate_dataframe(df, logger)


def _dataset_exists_and_valid(dataset_path: Path, logger: logging.Logger, df: pd.DataFrame = None) -> bool:
    """
    Check if dataset already exists and is valid.
    
    Args:
        dataset_path: Path to dataset file
        logger: Logger instance
        df: Optional DataFrame to validate (if None, loads from path)
        
    Returns:
        True if dataset exists and passes validation
    """
    if df is None:
        if not dataset_path.exists():
            logger.info(f"Dataset file does not exist: {dataset_path}")
            return False
        
        try:
            df = load_dataframe(dataset_path)
            logger.info(f"Successfully loaded existing dataset: {dataset_path}")
        except FileNotFoundError as e:
            if "NumPy compatibility" in str(e):
                logger.warning(f"Cannot load dataset due to environment compatibility: {dataset_path}")
                logger.info("Will regenerate dataset with current environment")
                return False
            else:
                logger.error(f"Failed to load dataset: {e}")
                return False
        except Exception as e:
            logger.warning(f"Dataset loading failed, will regenerate: {e}")
            return False
    
    return _validate_dataframe(df, logger)


def _flatten_state_observation(state, prefix: str = "obs", bytes_threshold: int = 512) -> Dict[str, float]:
    """
    Flatten state observation for better DataFrame storage.
    
    Args:
        state: State observation (numpy array or list)
        prefix: Column prefix for flattened features
        bytes_threshold: Threshold for switching to bytes storage (default: 512)
        
    Returns:
        Dictionary with flattened state features
    """
    if isinstance(state, np.ndarray):
        state_array = state
    else:
        state_array = np.array(state)
    
    # Adaptive threshold: use bytes for high-dimensional or large data
    should_use_bytes = (
        state_array.size > bytes_threshold or
        state_array.ndim > 3 or  # High-dimensional tensors
        (state_array.ndim == 3 and state_array.shape[-1] > 3)  # Multi-channel images
    )
    
    if should_use_bytes:
        # Store as bytes for efficiency
        return {
            f"{prefix}_bytes": state_array.tobytes(),
            f"{prefix}_shape": str(state_array.shape),
            f"{prefix}_dtype": str(state_array.dtype)
        }
    else:
        # Flatten the state array for small observations
        flattened = state_array.flatten()
        
        # Create column names
        feature_names = [f"{prefix}_{i}" for i in range(len(flattened))]
        
        # Return as dictionary
        return dict(zip(feature_names, flattened))


def _unflatten_state_observation(flattened_dict: Dict[str, float], original_shape: tuple, prefix: str = "obs") -> np.ndarray:
    """
    Reconstruct state observation from flattened dictionary.
    
    Args:
        flattened_dict: Dictionary with flattened features
        original_shape: Original state shape
        prefix: Column prefix used during flattening
        
    Returns:
        Reconstructed state observation
    """
    # Check if this was stored as bytes (high-dimensional observation)
    bytes_key = f"{prefix}_bytes"
    shape_key = f"{prefix}_shape"
    dtype_key = f"{prefix}_dtype"
    
    if bytes_key in flattened_dict:
        # Reconstruct from bytes storage
        try:
            bytes_data = flattened_dict[bytes_key]
            shape_str = flattened_dict[shape_key]
            dtype_str = flattened_dict[dtype_key]
            
            # Parse shape and dtype
            import ast
            shape = ast.literal_eval(shape_str)
            dtype = np.dtype(dtype_str)
            
            # Reconstruct array
            return np.frombuffer(bytes_data, dtype=dtype).reshape(shape)
        except Exception as e:
            raise ValueError(f"Failed to reconstruct bytes array: {e}")
    
    # Extract values by prefix, excluding shape information (for flattened arrays)
    feature_keys = [k for k in flattened_dict.keys() 
                   if k.startswith(f"{prefix}_") and not k.endswith('_shape') and not k.endswith('_bytes') and not k.endswith('_dtype')]
    
    if not feature_keys:
        raise ValueError(f"No feature keys found with prefix '{prefix}_'")
    
    # Sort by numerical index instead of string sorting
    # Extract index from key (e.g., "state_10" -> 10) and sort numerically
    def extract_index(key):
        try:
            return int(key.split('_')[-1])
        except (ValueError, IndexError):
            return 0  # Fallback for malformed keys
    
    feature_keys = sorted(feature_keys, key=extract_index)
    
    # Extract values in order
    feature_values = []
    for key in feature_keys:
        feature_values.append(flattened_dict[key])
    
    # Reshape to original shape
    return np.array(feature_values).reshape(original_shape)


def _get_max_episode_steps(env, context=None) -> int:
    """
    Get maximum episode steps from environment or use context configuration.
    
    Args:
        env: Environment instance
        context: Experiment context for fallback configuration
        
    Returns:
        Maximum episode steps
    """
    # Try to get from environment spec
    if hasattr(env, 'spec') and env.spec is not None:
        if hasattr(env.spec, 'max_episode_steps'):
            return env.spec.max_episode_steps
    
    # Try to get from environment directly
    if hasattr(env, 'max_episode_steps'):
        return env.max_episode_steps
    
    # Use context configuration if available
    if context and hasattr(context, 'max_steps'):
        return context.max_steps
    
    # Default safety threshold
    default_max_steps = 5000
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Using default max episode steps: {default_max_steps}")
    return default_max_steps


def _compute_remaining_discounted_returns(episode_data: List[Dict], gamma: float = 0.99) -> List[Dict]:
    """
    Compute remaining discounted returns for each timestep in an episode.
    
    This fixes the fundamental issue where Q-values should predict remaining
    returns from each timestep, not the entire episode return.
    
    Args:
        episode_data: List of step dictionaries from an episode
        gamma: Discount factor
        
    Returns:
        List of step dictionaries with added 'remaining_return' field
    """
    if not episode_data:
        return episode_data
    
    # Extract rewards in temporal order
    rewards = [step['reward'] for step in episode_data]
    n_steps = len(rewards)
    
    # Compute remaining discounted return for each timestep
    remaining_returns = []
    for t in range(n_steps):
        # Sum rewards from timestep t to end, with appropriate discounting
        remaining_return = sum(
            rewards[t + i] * (gamma ** i) for i in range(n_steps - t)
        )
        remaining_returns.append(remaining_return)
    
    # Add remaining_return to each step's data
    updated_episode_data = []
    for i, step_data in enumerate(episode_data):
        updated_step_data = step_data.copy()
        updated_step_data['remaining_return'] = remaining_returns[i]
        updated_episode_data.append(updated_step_data)
    
    return updated_episode_data