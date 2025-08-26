"""
Stage 1: Dataset Builder
Generate clean evaluation datasets for each environment type.

Migrated from analysis/modules/dataset_generator.py
Key functionality:
- Load trained models using stable-baselines3
- Generate evaluation episodes with deterministic policy
- Add Gaussian noise to observations based on env_type
- Save structured episode data as compressed DataFrame
"""

import logging
import re
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

# Stable-baselines3 imports (from original dataset_generator.py)
from stable_baselines3 import DQN
try:
    from sb3_contrib import QRDQN
    from rl_zoo3.bootstrapped_dqn import BootstrappedDQN
    from rl_zoo3.mcdropout_dqn import MCDropoutDQN
    SB3_AVAILABLE = True
except ImportError:
    # Fallback if specific UQ methods not available
    QRDQN = BootstrappedDQN = MCDropoutDQN = None
    SB3_AVAILABLE = False

try:
    from rl_zoo3.utils import get_saved_hyperparams, create_test_env
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
from ..utils.path_manager import get_clean_dataset_path, ensure_dir_exists
from ..utils.data_format import save_dataframe, load_dataframe, verify_file_integrity


def run(context: ExperimentContext) -> bool:
    """
    Stage 1: Generate clean evaluation datasets for all environment types.
    
    This stage:
    1. Iterates through all environment types (noise levels)
    2. Generates evaluation episodes for each environment configuration
    3. Saves clean datasets to standardized paths
    4. Validates generated datasets
    5. Supports resumption (skips existing valid datasets)
    
    Args:
        context: Experiment context with configuration
        
    Returns:
        True if all datasets generated successfully
    """
    logger = get_stage_logger("stage1_dataset_builder")
    
    with StageTimer(logger, "Dataset Generation") as timer:
        logger.info("=== Stage 1: Dataset Builder ===")
        
        # Check dependencies
        if not _check_dependencies(logger):
            return False
        
        env_types = context.env_types
        total_env_types = len(env_types)
        
        success_count = 0
        
        # Get available algorithms (only those with trained models)
        available_algorithms = _get_available_algorithms(context, logger)
        if not available_algorithms:
            logger.error("No trained models found for any configured algorithms")
            return False
        
        # Generate datasets for all algorithm-environment combinations
        total_combinations = len(available_algorithms) * len(env_types)
        logger.info(f"Will process {total_combinations} algorithm-environment combinations")
        
        start_time = time.time()
        combination_count = 0
        
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
                    continue
                
                # Generate dataset for this algorithm-environment combination
                env_start_time = time.time()
                success = _generate_dataset(context, env_type, dataset_path, logger, algorithm)
                env_duration = time.time() - env_start_time
                
                if success:
                    success_count += 1
                    logger.info(f"✓ Generated dataset for {algorithm}-{env_type} in {env_duration:.2f}s")
                    
                    # Memory usage info (optional)
                    try:
                        import psutil
                        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                        logger.debug(f"Memory usage: {memory_usage:.1f} MB")
                    except ImportError:
                        pass  # psutil not available, skip memory monitoring
                        
                else:
                    logger.error(f"✗ Failed to generate dataset for {algorithm}-{env_type} after {env_duration:.2f}s")
        
        total_duration = time.time() - start_time
        logger.info(f"Dataset generation completed: {success_count}/{total_combinations} successful in {total_duration:.2f}s")
        
        if success_count < total_combinations:
            logger.warning(f"Some datasets failed to generate. Success rate: {success_count/total_combinations*100:.1f}%")
        
        return success_count == total_combinations


def _get_available_algorithms(context: ExperimentContext, logger: logging.Logger) -> List[str]:
    """
    Check which algorithms have trained models available.
    
    Args:
        context: Experiment context
        logger: Logger instance
        
    Returns:
        List of algorithm names that have trained models
    """
    available_algorithms = []
    
    logger.info("Checking for available trained models...")
    
    for algorithm in context.algorithms:
        # Check if this algorithm has models for at least one environment type
        has_models = False
        
        for env_type in context.env_types:
            # Use first seed for checking model availability
            test_seed = context.seeds[0] if context.seeds else 101
            model_dir = Path(f"logs/multi_env_experiments/LunarLander-v3/{env_type}/{algorithm}/seed_{test_seed}_1")
            
            # Check for model files
            model_paths = [
                model_dir / "best_model.zip",
                model_dir / "LunarLander-v3.zip"
            ]
            
            if any(path.exists() for path in model_paths):
                has_models = True
                break
        
        if has_models:
            available_algorithms.append(algorithm)
            logger.info(f"✓ Found trained models for algorithm: {algorithm}")
        else:
            logger.warning(f"✗ No trained models found for algorithm: {algorithm} (skipping)")
    
    logger.info(f"Available algorithms: {available_algorithms}")
    return available_algorithms


def _check_dependencies(logger: logging.Logger) -> bool:
    """Check if all required dependencies are available."""
    if not SB3_AVAILABLE:
        logger.error("Stable-baselines3 UQ methods not available. Please install sb3-contrib.")
        return False
    
    if not RL_ZOO3_AVAILABLE:
        logger.error("RL-Zoo3 utilities not available. Please install rl-zoo3.")
        return False
    
    if not NOISE_WRAPPER_AVAILABLE:
        logger.error("Noise wrapper not available. Please check wrappers/noise.py.")
        return False
    
    return True


def _dataset_exists_and_valid(dataset_path: Path, logger: logging.Logger) -> bool:
    """
    Check if dataset already exists and is valid.
    
    Args:
        dataset_path: Path to dataset file
        logger: Logger instance
        
    Returns:
        True if dataset exists and passes validation
    """
    if not dataset_path.exists():
        return False
    
    try:
        # Try to load and validate the dataset
        df = load_dataframe(dataset_path)
        
        # Check required columns (from original dataset_generator.py analysis)
        required_columns = ['episode_id', 'step_id', 'state', 'action', 'reward', 'next_state', 'done']
        
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Dataset {dataset_path} missing required columns")
            return False
        
        # Check minimum data
        if len(df) < 10:  # Minimum threshold
            logger.warning(f"Dataset {dataset_path} has insufficient data: {len(df)} rows")
            return False
        
        logger.info(f"Valid dataset found: {dataset_path} ({len(df)} rows)")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to validate dataset {dataset_path}: {e}")
        return False


def _generate_dataset(context: ExperimentContext, env_type: str, 
                     dataset_path: Path, logger: logging.Logger, algorithm: str = "qrdqn") -> bool:
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
                                          algorithm: str = "qrdqn") -> Optional[pd.DataFrame]:
    """
    Generate episodes using trained QRDQN models for evaluation.
    
    Migrated from analysis/modules/dataset_generator.py load_evaluation_data() function.
    Key changes:
    - Uses trained QRDQN models instead of random policy
    - Loads models from logs/multi_env_experiments/LunarLander-v3/{env_type}/qrdqn/seed_{seed}_1/
    - Creates structured DataFrame output for UQ pipeline
    
    Args:
        context: Experiment context
        env_type: Environment type identifier
        noise_level: Noise level for environment
        logger: Logger instance
        
    Returns:
        DataFrame with episode data or None if failed
    """
    all_episode_data = []
    
    # Use first seed for dataset generation (we only need one representative dataset per env_type)
    test_seed = context.seeds[0] if context.seeds else 101
    
    logger.info(f"Loading trained {algorithm.upper()} model for {env_type}, seed: {test_seed}")
    
    try:
        # Build model path based on the directory structure we discovered
        model_dir = Path(f"logs/multi_env_experiments/LunarLander-v3/{env_type}/{algorithm}/seed_{test_seed}_1")
        
        # Try best_model.zip first, then fallback to LunarLander-v3.zip
        model_path = model_dir / "best_model.zip"
        if not model_path.exists():
            model_path = model_dir / "LunarLander-v3.zip"
            
        if not model_path.exists():
            logger.error(f"No trained model found at {model_dir}")
            return None
            
        logger.info(f"Loading model from: {model_path}")
        
        # Load the trained model based on algorithm type
        if algorithm == "qrdqn":
            model = QRDQN.load(str(model_path))
        elif algorithm == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(str(model_path))
        elif algorithm == "bootstrapped_dqn":
            # TODO: Import proper BootstrappedDQN when available
            model = QRDQN.load(str(model_path))  # Use QRDQN as placeholder
        elif algorithm == "mcdropout_dqn":
            # TODO: Import proper MCDropoutDQN when available  
            model = QRDQN.load(str(model_path))  # Use QRDQN as placeholder
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        logger.info(f"Successfully loaded {algorithm.upper()} model")
        
        # Create environment with noise (important: use same seed for consistency)
        env = _create_noisy_environment(context.env_id, noise_level, seed=test_seed)
        
        # Generate episodes with trained model policy
        episode_data = _collect_episode_data_trained_policy(env, model, context.eval_episodes, logger)
        
        env.close()
        
        if episode_data is not None and not episode_data.empty:
            # Add metadata columns
            episode_data['env_type'] = env_type
            episode_data['noise_level'] = noise_level
            episode_data['algorithm'] = algorithm
            episode_data['seed'] = test_seed
            
            all_episode_data.append(episode_data)
            
            logger.info(f"Generated {len(episode_data)} steps using trained QRDQN model")
        
    except Exception as e:
        logger.error(f"Failed to load model or generate episodes for {env_type}: {e}")
        logger.error(f"Model path attempted: {model_path if 'model_path' in locals() else 'unknown'}")
        return None
    
    if all_episode_data:
        return pd.concat(all_episode_data, ignore_index=True)
    else:
        return None


def _collect_episode_data_trained_policy(env, model, num_episodes: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
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
            max_steps = 1000  # Prevent infinite episodes
            
            while not done and step_id < max_steps:
                try:
                    # Store current state
                    current_state = obs.copy() if hasattr(obs, 'copy') else np.array(obs)
                    
                    # Use trained model to predict action (deterministic=True for evaluation)
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Ensure action is in correct format
                    if isinstance(action, np.ndarray):
                        action = action.item() if action.size == 1 else action[0]
                    
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
                        'state': current_state,
                        'action': int(action),  # Ensure it's an integer for discrete action space
                        'reward': float(reward),
                        'next_state': next_obs.copy() if hasattr(next_obs, 'copy') else np.array(next_obs),
                        'done': bool(done)
                    }
                    
                    episode_data.append(step_data)
                    
                    # Move to next step
                    obs = next_obs
                    step_id += 1
                    
                except Exception as e:
                    logger.warning(f"Step {step_id} failed in episode {episode_id}: {e}")
                    break  # Exit episode on step error
            
            # Episode completed
            if step_id > 0 and episode_data:
                all_data.extend(episode_data)
                successful_episodes += 1
                episode_duration = time.time() - episode_start_time
                logger.debug(f"Episode {episode_id}: {step_id} steps, final reward: {reward:.2f}, duration: {episode_duration:.2f}s")
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


def _collect_episode_data_random_policy(env, num_episodes: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Collect episode data using random policy.
    
    Migrated from analysis/modules/dataset_generator.py episode collection logic.
    Simplified version for demonstration - in practice, you'd use trained models.
    
    Args:
        env: Environment instance
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
            max_steps = 1000  # Prevent infinite episodes
            
            while not done and step_id < max_steps:
                try:
                    # Store current state
                    current_state = obs.copy() if hasattr(obs, 'copy') else np.array(obs)
                    
                    # Take random action (placeholder for trained model prediction)
                    action = env.action_space.sample()
                    
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
                    
                    # Store step data (format from original dataset_generator.py)
                    step_data = {
                        'episode_id': episode_id,
                        'step_id': step_id,
                        'state': current_state,
                        'action': action,
                        'reward': float(reward),  # Ensure it's a Python float
                        'next_state': next_obs.copy() if hasattr(next_obs, 'copy') else np.array(next_obs),
                        'done': bool(done)  # Ensure it's a Python bool
                    }
                    
                    episode_data.append(step_data)
                    
                    # Move to next step
                    obs = next_obs
                    step_id += 1
                    
                except Exception as e:
                    logger.warning(f"Step {step_id} failed in episode {episode_id}: {e}")
                    break  # Exit episode on step error
            
            # Episode completed
            if step_id > 0 and episode_data:
                all_data.extend(episode_data)
                successful_episodes += 1
                episode_duration = time.time() - episode_start_time
                logger.debug(f"Episode {episode_id}: {step_id} steps, final reward: {reward:.2f}, duration: {episode_duration:.2f}s")
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
    return _generate_episodes_with_trained_models(context, env_type, noise_level, logger)


def _collect_episode_data(env, num_episodes: int, logger: logging.Logger) -> pd.DataFrame:
    """Legacy wrapper - calls the new implementation."""
    return _collect_episode_data_random_policy(env, num_episodes, logger)


def validate_dataset_structure(df: pd.DataFrame) -> bool:
    """
    Validate that dataset has required structure and columns.
    
    Args:
        df: Dataset DataFrame to validate
        
    Returns:
        True if dataset structure is valid
    """
    required_columns = [
        'episode_id',
        'step_id', 
        'state',
        'action',
        'reward',
        'next_state',
        'done'
    ]
    
    # Check required columns exist
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for missing values in critical columns
    critical_columns = ['episode_id', 'step_id', 'action', 'reward', 'done']
    for col in critical_columns:
        if df[col].isnull().any():
            print(f"Found null values in critical column: {col}")
            return False
    
    # Check episode continuity (episodes should have consecutive step_ids)
    for episode_id in df['episode_id'].unique():
        episode_data = df[df['episode_id'] == episode_id].sort_values('step_id')
        expected_steps = list(range(len(episode_data)))
        actual_steps = episode_data['step_id'].tolist()
        if actual_steps != expected_steps:
            print(f"Episode {episode_id} has non-consecutive step_ids")
            return False
    
    return True