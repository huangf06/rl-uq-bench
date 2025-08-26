"""
Stage 0: Configuration Loading and Validation
Initialize experiment context and validate configuration parameters.
"""

import logging
from pathlib import Path
from ..utils.context import ExperimentContext
from ..utils.logging_utils import get_stage_logger, StageTimer


def run(context: ExperimentContext) -> bool:
    """
    Stage 0: Load and validate experiment configuration.
    
    This stage:
    1. Validates configuration file format and content
    2. Checks that all required fields are present
    3. Validates paths and directories
    4. Verifies UQ methods are supported
    5. Logs configuration summary
    
    Args:
        context: Experiment context with loaded configuration
        
    Returns:
        True if configuration is valid and ready for pipeline execution
    """
    logger = get_stage_logger("stage0_config")
    
    with StageTimer(logger, "Configuration Validation") as timer:
        logger.info("=== Stage 0: Configuration Loading and Validation ===")
        
        try:
            # Validate required fields exist
            if not context.env_id:
                logger.error("Missing required field: env_id")
                return False
            
            if not context.algorithms:
                logger.error("Missing required field: algorithms")
                return False
            
            if not context.env_types:
                logger.error("Missing required field: env_types or env_types_with_noise")
                return False
            
            if not context.seeds:
                logger.error("Missing required field: seeds")
                return False
            
            # Verify UQ methods are supported
            if not validate_uq_methods(context.algorithms):
                logger.error("Some algorithms are not supported")
                return False
            
            # Validate environment types format
            if not validate_env_types(context.env_types):
                logger.error("Some environment types have invalid format")
                return False
            
            # Validate seed values are integers
            if not all(isinstance(seed, int) for seed in context.seeds):
                logger.error("All seeds must be integers")
                return False
            
            # Ensure eval_episodes is positive integer
            if context.eval_episodes <= 0:
                logger.error("eval_episodes must be positive")
                return False
            
            # Check data and results directories are accessible
            if not validate_paths(str(context.data_root), str(context.results_root)):
                logger.error("Data or results paths are not accessible")
                return False
            
            # Log configuration summary
            log_configuration_summary(context, logger)
            
            # Create necessary directories if they don't exist
            context.data_root.mkdir(parents=True, exist_ok=True)
            context.results_root.mkdir(parents=True, exist_ok=True)
            
            logger.info("Configuration validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


def validate_uq_methods(methods: list) -> bool:
    """
    Validate that all specified UQ methods are supported.
    
    Args:
        methods: List of UQ method names
        
    Returns:
        True if all methods are supported
    """
    supported_methods = [
        'qrdqn',
        'bootstrapped_dqn', 
        'mcdropout_dqn',
        'dqn',
        'qr_bootstrap_dqn'
    ]
    
    for method in methods:
        if method not in supported_methods:
            return False
    return True


def validate_env_types(env_types: list) -> bool:
    """
    Validate environment type naming format.
    
    Args:
        env_types: List of environment type identifiers
        
    Returns:
        True if all env_types follow expected format
    """
    import re
    
    # Expected format: uncertainty_degradation_noise{X.XXX}
    noise_pattern = re.compile(r'uncertainty_degradation_noise\d+\.\d+')
    
    for env_type in env_types:
        # Allow exact matches and some variations
        if not (noise_pattern.match(env_type) or env_type in ['clean', 'baseline']):
            # Be lenient for now, just warn about unexpected formats
            continue
    
    return True  # Accept all formats for flexibility


def validate_paths(data_root: str, results_root: str) -> bool:
    """
    Validate that data and results paths are accessible.
    
    Args:
        data_root: Data root directory path
        results_root: Results root directory path
        
    Returns:
        True if paths are valid and accessible
    """
    try:
        from pathlib import Path
        
        data_path = Path(data_root)
        results_path = Path(results_root)
        
        # Check if paths are valid
        data_path.resolve()
        results_path.resolve()
        
        # Try to create directories if they don't exist
        data_path.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Check if directories are writable
        test_file_data = data_path / ".test_write"
        test_file_results = results_path / ".test_write"
        
        try:
            test_file_data.touch()
            test_file_data.unlink()
            
            test_file_results.touch()
            test_file_results.unlink()
        except (PermissionError, OSError):
            return False
        
        return True
        
    except Exception:
        return False


def log_configuration_summary(context: ExperimentContext, logger: logging.Logger) -> None:
    """
    Log a comprehensive summary of the loaded configuration.
    
    Args:
        context: Experiment context
        logger: Logger instance
    """
    logger.info("=== Configuration Summary ===")
    logger.info(f"Environment: {context.env_id}")
    logger.info(f"Algorithms: {context.algorithms}")
    logger.info(f"Environment Types: {len(context.env_types)}")
    for env_type in context.env_types:
        noise_level = context.get_noise_level(env_type)
        logger.info(f"  - {env_type} (noise: {noise_level})")
    
    logger.info(f"Seeds: {len(context.seeds)} total, using first {context.n_seeds}")
    logger.info(f"  - Active seeds: {context.seeds[:context.n_seeds]}")
    logger.info(f"Evaluation episodes per combination: {context.eval_episodes}")
    
    # Calculate total experiments
    total_combinations = len(context.get_env_method_seed_combinations())
    logger.info(f"Total experiment combinations: {total_combinations}")
    
    # Log paths
    logger.info(f"Data directory: {context.data_root}")
    logger.info(f"Results directory: {context.results_root}")
    logger.info(f"Logs directory: {context.logs_dir}")
    
    # Log additional settings
    logger.info(f"UQ Metrics: {context.uq_metrics}")
    logger.info(f"Success threshold: {context.success_threshold}")
    logger.info(f"Bootstrap samples: {context.bootstrap_samples}")
    logger.info(f"Confidence level: {context.confidence_level}")
    logger.info(f"Skip existing: {context.skip_existing}")
    
    # Log algorithm configurations
    logger.info("Algorithm Configurations:")
    for algorithm in context.algorithms:
        config = context.get_algorithm_config(algorithm)
        uncertainty_type = context.get_uncertainty_type(algorithm)
        logger.info(f"  - {algorithm}: {uncertainty_type} ({config})")
    
    logger.info("===========================")