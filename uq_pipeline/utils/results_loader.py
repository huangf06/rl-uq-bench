"""
UQ Results Loader - New Architecture Data Access

Helper functions to load and analyze UQ metrics from the new two-tier architecture.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from ..utils.context import ExperimentContext
from ..utils.path_manager import get_metrics_episode_path


def load_episode_metrics(context: ExperimentContext, 
                        env_type: Optional[str] = None,
                        method: Optional[str] = None,
                        seed: Optional[int] = None) -> pd.DataFrame:
    """
    Load episode-level UQ metrics from the new Stage 4a.
    
    Args:
        context: Experiment context
        env_type: Environment type filter (optional)
        method: Method filter (optional) 
        seed: Seed filter (optional)
        
    Returns:
        DataFrame with episode-level metrics
    """
    results = []
    
    combinations = context.get_env_method_seed_combinations()
    
    for comb_env_type, comb_method, comb_seed in combinations:
        # Apply filters
        if env_type and comb_env_type != env_type:
            continue
        if method and comb_method != method:
            continue
        if seed and comb_seed != seed:
            continue
            
        # Load episode metrics
        episode_path = get_metrics_episode_path(
            context.results_root, context.env_id, comb_env_type, comb_method, comb_seed
        )
        
        if episode_path.exists():
            df = pd.read_csv(episode_path)
            df['env_type'] = comb_env_type
            df['noise_level'] = float(comb_env_type.split('noise')[-1])
            results.append(df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def get_core_metrics_summary(context: ExperimentContext) -> Dict[str, pd.DataFrame]:
    """
    Get summary statistics for the 6 core UQ metrics.
    
    Returns:
        Dictionary with summary stats by method and noise level
    """
    df = load_episode_metrics(context)
    
    if len(df) == 0:
        return {}
    
    core_metrics = ['crps', 'wis', 'ace', 'picp_50', 'picp_90', 'interval_width']
    
    summaries = {}
    
    # Overall summary
    summaries['overall'] = df[core_metrics].describe()
    
    # By method
    if 'algorithm' in df.columns:
        summaries['by_method'] = df.groupby('algorithm')[core_metrics].mean()
    
    # By noise level  
    if 'noise_level' in df.columns:
        summaries['by_noise'] = df.groupby('noise_level')[core_metrics].mean()
    
    # By method and noise
    if 'algorithm' in df.columns and 'noise_level' in df.columns:
        summaries['by_method_noise'] = df.groupby(['algorithm', 'noise_level'])[core_metrics].mean()
    
    return summaries


def compare_methods_on_metric(context: ExperimentContext, metric: str = 'crps') -> pd.DataFrame:
    """
    Compare methods on a specific UQ metric across noise levels.
    
    Args:
        context: Experiment context
        metric: Metric to compare ('crps', 'wis', 'ace', etc.)
        
    Returns:
        DataFrame with method comparison
    """
    df = load_episode_metrics(context)
    
    if len(df) == 0 or metric not in df.columns:
        return pd.DataFrame()
    
    # Aggregate by method and noise level
    comparison = df.groupby(['algorithm', 'noise_level'])[metric].agg(['mean', 'std', 'count']).reset_index()
    
    # Pivot for easier comparison
    mean_comparison = comparison.pivot(index='algorithm', columns='noise_level', values='mean')
    
    return mean_comparison


def get_calibration_analysis(context: ExperimentContext) -> Dict[str, pd.DataFrame]:
    """
    Analyze calibration performance across methods and noise levels.
    
    Returns:
        Dictionary with calibration analysis
    """
    df = load_episode_metrics(context)
    
    if len(df) == 0:
        return {}
    
    calibration_metrics = ['ace', 'picp_50', 'picp_90']
    
    analysis = {}
    
    # Perfect calibration targets
    perfect_picp_50 = 0.5
    perfect_picp_90 = 0.9
    perfect_ace = 0.0
    
    # Calculate calibration errors
    if all(metric in df.columns for metric in calibration_metrics):
        df_cal = df.copy()
        df_cal['picp_50_error'] = abs(df_cal['picp_50'] - perfect_picp_50)
        df_cal['picp_90_error'] = abs(df_cal['picp_90'] - perfect_picp_90)
        df_cal['ace_score'] = df_cal['ace']  # ACE is already an error measure
        
        # Summary by method and noise
        analysis['calibration_errors'] = df_cal.groupby(['algorithm', 'noise_level'])[
            ['picp_50_error', 'picp_90_error', 'ace_score']
        ].mean()
        
        # Overall calibration ranking
        analysis['method_ranking'] = df_cal.groupby('algorithm')[
            ['picp_50_error', 'picp_90_error', 'ace_score']
        ].mean().sort_values('ace_score')
    
    return analysis


def export_results_for_analysis(context: ExperimentContext, output_path: str = "uq_analysis_results.csv"):
    """
    Export all episode-level results to a single CSV for external analysis.
    
    Args:
        context: Experiment context
        output_path: Output file path
    """
    df = load_episode_metrics(context)
    
    if len(df) > 0:
        output_file = Path(output_path)
        df.to_csv(output_file, index=False)
        print(f"✅ Results exported to {output_file}")
        print(f"📊 Total episodes: {len(df)}")
        print(f"🎯 Core metrics: {[col for col in df.columns if col in ['crps', 'wis', 'ace', 'picp_50', 'picp_90', 'interval_width']]}")
        return str(output_file)
    else:
        print("❌ No data found to export")
        return None


# Quick analysis functions
def quick_performance_summary(context: ExperimentContext):
    """Print a quick performance summary of all methods."""
    summaries = get_core_metrics_summary(context)
    
    print("🎯 UQ Performance Summary (New Architecture)")
    print("=" * 50)
    
    if 'by_method_noise' in summaries:
        print("\n📊 Performance by Method and Noise Level:")
        print(summaries['by_method_noise'][['crps', 'wis', 'ace']].round(4))
    
    if 'by_method' in summaries:
        print(f"\n🏆 Overall Method Ranking (by CRPS):")
        method_ranking = summaries['by_method']['crps'].sort_values()
        for i, (method, score) in enumerate(method_ranking.items(), 1):
            print(f"  {i}. {method}: {score:.4f}")
    
    print(f"\n✅ Analysis based on {len(load_episode_metrics(context))} episodes")
    print("📈 Using new two-tier architecture with 6 core metrics")