"""
UQ Pipeline Runner
Main entry point for the uncertainty quantification evaluation pipeline.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from .utils.context import ExperimentContext
from .utils.logging_utils import setup_logger, StageTimer
from .stages import (
    stage0_config,
    stage1_dataset_builder,
    stage2_performance,
    stage3_q_extractor,
    stage4_metrics,
    stage5_calibration,
    stage6_report
)


def main():
    """
    Main entry point for UQ pipeline execution.
    
    Supports:
    - Running all stages sequentially
    - Running specific stages only
    - Resuming from interrupted runs
    - Parallel execution where applicable
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger("uq_pipeline", args.log_file, 
                         logging.DEBUG if args.verbose else logging.INFO)
    
    try:
        # Initialize experiment context
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        context = ExperimentContext(config_path)
        
        # Determine which stages to run
        stages_to_run = _determine_stages_to_run(args, logger)
        
        # Execute pipeline
        success = run_pipeline(context, stages_to_run, args, logger)
        
        if success:
            logger.info("=== UQ Pipeline completed successfully ===")
            return 0
        else:
            logger.error("=== UQ Pipeline completed with errors ===")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        if args.verbose:
            logger.exception("Full error details:")
        return 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="UQ Pipeline: Uncertainty Quantification Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml
  
  # Run specific stages
  python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml --stages 1,2,3
  
  # Resume from stage 3
  python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml --from-stage 3
  
  # Verbose logging
  python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    # Stage selection arguments
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        '--stages',
        type=str,
        help='Comma-separated list of stages to run (0-6)'
    )
    stage_group.add_argument(
        '--from-stage',
        type=int,
        choices=range(7),
        help='Run from specified stage to end (0-6)'
    )
    stage_group.add_argument(
        '--only-stage',
        type=int,
        choices=range(7),
        help='Run only the specified stage (0-6)'
    )
    
    # Execution options
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run (skip completed experiments)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run all experiments (ignore existing results)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and show execution plan without running'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (default: logs to console)'
    )
    
    # Performance options
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel execution where possible'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    return parser.parse_args()


def _determine_stages_to_run(args: argparse.Namespace, logger: logging.Logger) -> List[int]:
    """
    Determine which stages to run based on arguments.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
        
    Returns:
        List of stage numbers to execute
    """
    all_stages = list(range(7))  # Stages 0-6
    
    if args.stages:
        # Parse comma-separated stage list
        try:
            stages = [int(s.strip()) for s in args.stages.split(',')]
            stages = [s for s in stages if 0 <= s <= 6]
            return sorted(stages)
        except ValueError:
            logger.error("Invalid stages format. Use comma-separated integers (0-6)")
            return []
    
    elif args.from_stage is not None:
        # Run from specified stage to end
        return list(range(args.from_stage, 7))
    
    elif args.only_stage is not None:
        # Run only specified stage
        return [args.only_stage]
    
    else:
        # Run all stages
        return all_stages


def run_pipeline(context: ExperimentContext, stages_to_run: List[int], 
                args: argparse.Namespace, logger: logging.Logger) -> bool:
    """
    Execute the UQ pipeline stages.
    
    Args:
        context: Experiment context
        stages_to_run: List of stage numbers to execute
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        True if all stages completed successfully
    """
    # Stage definitions
    stage_definitions = {
        0: ("Configuration Validation", stage0_config.run),
        1: ("Dataset Builder", stage1_dataset_builder.run),
        2: ("Performance Evaluation", stage2_performance.run),
        3: ("Q-value Extraction", stage3_q_extractor.run),
        4: ("Raw Metrics Computation", stage4_metrics.run),
        5: ("Calibration", stage5_calibration.run),
        6: ("Report Generation", stage6_report.run)
    }
    
    with StageTimer(logger, "Complete Pipeline") as pipeline_timer:
        
        # Dry run mode
        if args.dry_run:
            return _execute_dry_run(context, stages_to_run, stage_definitions, logger)
        
        # Execute stages
        for stage_num in stages_to_run:
            if stage_num not in stage_definitions:
                logger.error(f"Invalid stage number: {stage_num}")
                return False
            
            stage_name, stage_func = stage_definitions[stage_num]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Stage {stage_num}: {stage_name}")
            logger.info(f"{'='*60}")
            
            try:
                with StageTimer(logger, f"Stage {stage_num}") as stage_timer:
                    # TODO: Set resume mode in context if needed
                    if args.resume:
                        context.resume_mode = True
                    if args.force:
                        context.force_mode = True
                    
                    success = stage_func(context)
                
                if success:
                    logger.info(f"Stage {stage_num} completed successfully in {stage_timer.duration:.2f}s")
                else:
                    logger.error(f"Stage {stage_num} failed")
                    return False
                    
            except Exception as e:
                logger.error(f"Stage {stage_num} failed with exception: {e}")
                if args.verbose:
                    logger.exception("Full error details:")
                return False
        
        logger.info(f"\nPipeline completed in {pipeline_timer.duration:.2f}s")
        return True


def _execute_dry_run(context: ExperimentContext, stages_to_run: List[int], 
                    stage_definitions: dict, logger: logging.Logger) -> bool:
    """
    Execute dry run mode to validate configuration and show execution plan.
    
    Args:
        context: Experiment context
        stages_to_run: List of stage numbers
        stage_definitions: Stage definitions dictionary
        logger: Logger instance
        
    Returns:
        True if dry run validation passed
    """
    logger.info("=== DRY RUN MODE ===")
    
    # TODO: Validate configuration
    try:
        # Run stage 0 (configuration validation) if included
        if 0 in stages_to_run:
            success = stage0_config.run(context)
            if not success:
                logger.error("Configuration validation failed")
                return False
        
        # Show execution plan
        logger.info("\nExecution Plan:")
        logger.info("-" * 40)
        
        total_combinations = len(context.get_env_method_seed_combinations())
        
        for stage_num in stages_to_run:
            stage_name, _ = stage_definitions[stage_num]
            logger.info(f"Stage {stage_num}: {stage_name}")
            
            if stage_num > 0:  # Skip stage 0 for combination counting
                logger.info(f"  - Will process {total_combinations} experiment combinations")
        
        logger.info(f"\nTotal experiment combinations: {total_combinations}")
        logger.info(f"Environment types: {len(context.env_types)}")
        logger.info(f"UQ methods: {len(context.uq_methods)}")
        logger.info(f"Seeds: {len(context.seeds)}")
        
        # TODO: Estimate storage and runtime
        estimated_storage, estimated_runtime = _estimate_requirements(context, stages_to_run)
        logger.info(f"\nEstimated storage required: {estimated_storage:.2f} GB")
        logger.info(f"Estimated runtime: {estimated_runtime:.1f} hours")
        
        logger.info("\nDry run completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Dry run failed: {e}")
        return False


def _estimate_requirements(context: ExperimentContext, stages_to_run: List[int]) -> tuple:
    """
    Estimate storage and runtime requirements.
    
    Args:
        context: Experiment context
        stages_to_run: List of stage numbers
        
    Returns:
        Tuple of (estimated_storage_gb, estimated_runtime_hours)
    """
    # TODO: Implement requirement estimation
    # - Estimate based on number of combinations
    # - Account for Q-value array sizes
    # - Consider compression ratios
    # - Estimate processing time per combination
    
    num_combinations = len(context.get_env_method_seed_combinations())
    
    # Rough estimates (to be refined based on actual data)
    storage_per_combination_mb = 100  # Placeholder
    runtime_per_combination_minutes = 5  # Placeholder
    
    estimated_storage_gb = (num_combinations * storage_per_combination_mb) / 1024
    estimated_runtime_hours = (num_combinations * runtime_per_combination_minutes) / 60
    
    return estimated_storage_gb, estimated_runtime_hours


def show_pipeline_status(config_path: Path) -> None:
    """
    Show current pipeline execution status.
    
    Args:
        config_path: Path to configuration file
    """
    # TODO: Implement status checking
    # - Load context
    # - Check completed experiments
    # - Show progress summary
    # - Identify failed/incomplete runs
    pass


if __name__ == "__main__":
    sys.exit(main())