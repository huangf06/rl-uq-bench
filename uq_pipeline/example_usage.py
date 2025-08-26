#!/usr/bin/env python3
"""
UQ Pipeline Usage Examples
Demonstrates how to use the UQ pipeline system programmatically.
"""

import sys
from pathlib import Path

# Add uq_pipeline to Python path
sys.path.append(str(Path(__file__).parent.parent))

from uq_pipeline import ExperimentContext, run_pipeline
from uq_pipeline.utils import setup_logger


def example_basic_usage():
    """
    Basic usage example: Load configuration and run pipeline.
    """
    print("=== Basic Usage Example ===")
    
    # Setup logging
    logger = setup_logger("example", level="INFO")
    
    # Load experiment configuration
    config_path = Path("uq_pipeline/configs/experiment_lunarlander.yml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    try:
        # Initialize experiment context
        context = ExperimentContext(config_path)
        
        # Show configuration summary
        logger.info(f"Environment: {context.env_id}")
        logger.info(f"UQ Methods: {context.uq_methods}")
        logger.info(f"Environment Types: {len(context.env_types)}")
        logger.info(f"Seeds: {context.seeds}")
        
        # Get all experiment combinations
        combinations = context.get_env_method_seed_combinations()
        logger.info(f"Total experiment combinations: {len(combinations)}")
        
        # Example: Show first few combinations
        for i, (env_type, method, seed) in enumerate(combinations[:5]):
            logger.info(f"  {i+1}. {env_type} / {method} / seed_{seed}")
        
        return True
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        return False


def example_stage_execution():
    """
    Example: Running specific pipeline stages.
    """
    print("\n=== Stage Execution Example ===")
    
    logger = setup_logger("stage_example", level="INFO")
    config_path = Path("uq_pipeline/configs/experiment_lunarlander.yml")
    
    try:
        context = ExperimentContext(config_path)
        
        # Example: Simulate running stage 0 (config validation)
        from uq_pipeline.stages import stage0_config
        
        logger.info("Running Stage 0: Configuration Validation")
        # success = stage0_config.run(context)
        # logger.info(f"Stage 0 result: {'Success' if success else 'Failed'}")
        
        # NOTE: Since the actual implementation is not complete,
        # we'll just show the interface
        logger.info("Stage 0 interface ready - implementation needed")
        
        return True
        
    except Exception as e:
        logger.error(f"Stage execution example failed: {e}")
        return False


def example_path_management():
    """
    Example: Using path management utilities.
    """
    print("\n=== Path Management Example ===")
    
    logger = setup_logger("path_example", level="INFO")
    
    try:
        from uq_pipeline.utils.path_manager import (
            get_clean_dataset_path, get_result_dir, get_q_values_path
        )
        
        # Example paths
        data_root = Path("uq_results/data")
        results_root = Path("uq_results/results")
        env_id = "LunarLander-v3"
        env_type = "uncertainty_degradation_noise0.050"
        method = "qrdqn"
        seed = 101
        
        # Show path generation examples
        # dataset_path = get_clean_dataset_path(data_root, env_id, env_type)
        # result_dir = get_result_dir(results_root, env_id, env_type, method, seed)
        # q_values_path = get_q_values_path(results_root, env_id, env_type, method, seed)
        
        logger.info("Path management utilities ready")
        logger.info(f"Data root: {data_root}")
        logger.info(f"Results root: {results_root}")
        logger.info("Path generation functions available")
        
        return True
        
    except Exception as e:
        logger.error(f"Path management example failed: {e}")
        return False


def example_data_format():
    """
    Example: Using data format utilities.
    """
    print("\n=== Data Format Example ===")
    
    logger = setup_logger("data_example", level="INFO")
    
    try:
        from uq_pipeline.utils.data_format import (
            save_json, load_json, save_dataframe, load_dataframe
        )
        import pandas as pd
        
        # Example data
        example_data = {
            "experiment": "test",
            "method": "qrdqn", 
            "performance": 0.85
        }
        
        example_df = pd.DataFrame({
            "metric": ["uncertainty", "confidence", "accuracy"],
            "value": [0.12, 0.88, 0.92]
        })
        
        logger.info("Data format utilities ready")
        logger.info(f"Example JSON data: {example_data}")
        logger.info(f"Example DataFrame shape: {example_df.shape}")
        logger.info("Save/load functions available for JSON, CSV, compressed formats")
        
        return True
        
    except Exception as e:
        logger.error(f"Data format example failed: {e}")
        return False


def main():
    """
    Run all usage examples.
    """
    print("UQ Pipeline Usage Examples")
    print("=" * 50)
    
    # Run examples
    examples = [
        example_basic_usage,
        example_stage_execution,
        example_path_management,
        example_data_format
    ]
    
    results = []
    for example_func in examples:
        try:
            result = example_func()
            results.append(result)
        except Exception as e:
            print(f"Example {example_func.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Examples Summary:")
    successful = sum(results)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    
    if successful == total:
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Implement the TODO functions in each module")
        print("2. Add your specific UQ methods and environments")
        print("3. Run the actual pipeline with your data")
    else:
        print("⚠️  Some examples had issues - check implementations")
    
    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)