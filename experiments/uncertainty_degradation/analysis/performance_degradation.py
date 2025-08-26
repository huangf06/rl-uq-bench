#!/usr/bin/env python3
"""
QR-DQN 不确定性退化分析 - 性能分析模块
分析不同噪声水平下的性能退化趋势
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json

class PerformanceDegradationAnalyzer:
    def __init__(self, base_path="logs/multi_env_experiments/LunarLander-v3"):
        self.base_path = Path(base_path)
        self.noise_levels = [0.000, 0.025, 0.050, 0.075, 0.100]
        self.seeds = [101, 307, 911, 1747, 2029, 2861, 3253, 4099, 7919, 9011]
        self.results = {}
        
    def load_experiment_results(self, noise_level):
        """加载指定噪声水平的实验结果"""
        experiment_note = f"uncertainty_degradation_noise{noise_level:.3f}".replace('.', '_')
        exp_path = self.base_path / experiment_note / "qrdqn"
        
        if not exp_path.exists():
            print(f"Warning: Experiment path not found: {exp_path}")
            return None
            
        results = []
        for seed in self.seeds:
            seed_path = exp_path / f"seed_{seed}_1"
            eval_file = seed_path / "evaluations.npz"
            
            if eval_file.exists():
                try:
                    data = np.load(eval_file)
                    final_reward = data['results'].mean()  # 取最终评估的平均分数
                    results.append({
                        'seed': seed,
                        'noise_level': noise_level,
                        'final_reward': final_reward
                    })
                except Exception as e:
                    print(f"Error loading {eval_file}: {e}")
            else:
                print(f"Warning: Evaluation file not found: {eval_file}")
                
        return results
    
    def analyze_all_experiments(self):
        """分析所有噪声水平的实验结果"""
        all_results = []
        
        for noise_level in self.noise_levels:
            print(f"Loading results for noise level {noise_level}...")
            results = self.load_experiment_results(noise_level)
            if results:
                all_results.extend(results)
                self.results[noise_level] = results
            else:
                print(f"No results found for noise level {noise_level}")
        
        # 转换为DataFrame
        self.df = pd.DataFrame(all_results)
        return self.df
    
    def calculate_degradation_metrics(self):
        """计算退化指标"""
        if self.df.empty:
            print("No data available for analysis")
            return None
            
        metrics = []
        for noise_level in self.noise_levels:
            noise_data = self.df[self.df['noise_level'] == noise_level]
            if not noise_data.empty:
                metrics.append({
                    'noise_level': noise_level,
                    'mean_reward': noise_data['final_reward'].mean(),
                    'std_reward': noise_data['final_reward'].std(),
                    'min_reward': noise_data['final_reward'].min(),
                    'max_reward': noise_data['final_reward'].max(),
                    'success_rate': (noise_data['final_reward'] >= 200).mean(),
                    'usable_rate': (noise_data['final_reward'] >= 150).mean(),
                    'n_runs': len(noise_data)
                })
        
        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df
    
    def plot_degradation_trends(self, save_path="experiments/uncertainty_degradation/results"):
        """绘制退化趋势图"""
        if self.metrics_df is None:
            print("No metrics available for plotting")
            return
            
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 平均分数退化趋势
        axes[0, 0].plot(self.metrics_df['noise_level'], self.metrics_df['mean_reward'], 
                        'o-', linewidth=2, markersize=8, color='blue')
        axes[0, 0].fill_between(self.metrics_df['noise_level'], 
                                self.metrics_df['mean_reward'] - self.metrics_df['std_reward'],
                                self.metrics_df['mean_reward'] + self.metrics_df['std_reward'],
                                alpha=0.3, color='blue')
        axes[0, 0].set_xlabel('Noise Level (σ)')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].set_title('Performance Degradation Trend')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率退化趋势
        axes[0, 1].plot(self.metrics_df['noise_level'], self.metrics_df['success_rate'], 
                        's-', linewidth=2, markersize=8, color='green', label='Success Rate (≥200)')
        axes[0, 1].plot(self.metrics_df['noise_level'], self.metrics_df['usable_rate'], 
                        '^-', linewidth=2, markersize=8, color='orange', label='Usable Rate (≥150)')
        axes[0, 1].set_xlabel('Noise Level (σ)')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_title('Success Rate Degradation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 分数分布箱线图
        box_data = []
        box_labels = []
        for noise_level in self.noise_levels:
            noise_data = self.df[self.df['noise_level'] == noise_level]['final_reward']
            if not noise_data.empty:
                box_data.append(noise_data.values)
                box_labels.append(f'σ={noise_level}')
        
        axes[1, 0].boxplot(box_data, labels=box_labels)
        axes[1, 0].set_xlabel('Noise Level (σ)')
        axes[1, 0].set_ylabel('Final Reward')
        axes[1, 0].set_title('Reward Distribution by Noise Level')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 标准差趋势
        axes[1, 1].plot(self.metrics_df['noise_level'], self.metrics_df['std_reward'], 
                        'd-', linewidth=2, markersize=8, color='red')
        axes[1, 1].set_xlabel('Noise Level (σ)')
        axes[1, 1].set_ylabel('Standard Deviation')
        axes[1, 1].set_title('Consistency Degradation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'performance_degradation_trends.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'performance_degradation_trends.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"Plots saved to {save_path}")
    
    def save_results(self, save_path="experiments/uncertainty_degradation/results"):
        """保存分析结果"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        if not self.df.empty:
            self.df.to_csv(save_path / 'detailed_results.csv', index=False)
            print(f"Detailed results saved to {save_path / 'detailed_results.csv'}")
        
        # 保存汇总指标
        if self.metrics_df is not None:
            self.metrics_df.to_csv(save_path / 'performance_summary.csv', index=False)
            print(f"Performance summary saved to {save_path / 'performance_summary.csv'}")
            
            # 保存为JSON格式，便于后续分析
            summary_dict = {
                'experiment_info': {
                    'noise_levels': self.noise_levels,
                    'seeds': self.seeds,
                    'total_runs': len(self.df) if not self.df.empty else 0
                },
                'performance_metrics': self.metrics_df.to_dict('records')
            }
            
            with open(save_path / 'performance_summary.json', 'w') as f:
                json.dump(summary_dict, f, indent=2)
            print(f"Performance summary (JSON) saved to {save_path / 'performance_summary.json'}")
    
    def generate_report(self):
        """生成分析报告"""
        if self.metrics_df is None:
            print("No metrics available for report generation")
            return
            
        print("\n" + "="*60)
        print("QR-DQN 不确定性退化分析报告")
        print("="*60)
        
        print(f"\n实验设置:")
        print(f"- 噪声水平: {self.noise_levels}")
        print(f"- 随机种子数量: {len(self.seeds)}")
        print(f"- 总实验次数: {len(self.df) if not self.df.empty else 0}")
        
        print(f"\n性能退化趋势:")
        for _, row in self.metrics_df.iterrows():
            print(f"σ={row['noise_level']:.3f}: "
                  f"平均分={row['mean_reward']:.1f}±{row['std_reward']:.1f}, "
                  f"成功率={row['success_rate']:.1%}, "
                  f"可用率={row['usable_rate']:.1%}")
        
        # 计算退化率
        if len(self.metrics_df) > 1:
            baseline = self.metrics_df.iloc[0]['mean_reward']
            final = self.metrics_df.iloc[-1]['mean_reward']
            degradation_rate = (baseline - final) / baseline * 100 if baseline > 0 else 0
            print(f"\n总体退化率: {degradation_rate:.1f}% (从σ=0.000到σ=0.100)")
        
        print("="*60)

def main():
    """主函数"""
    analyzer = PerformanceDegradationAnalyzer()
    
    # 加载所有实验结果
    print("Loading experiment results...")
    df = analyzer.analyze_all_experiments()
    
    if df is not None and not df.empty:
        # 计算退化指标
        print("Calculating degradation metrics...")
        metrics = analyzer.calculate_degradation_metrics()
        
        # 生成报告
        analyzer.generate_report()
        
        # 绘制趋势图
        print("Generating plots...")
        analyzer.plot_degradation_trends()
        
        # 保存结果
        print("Saving results...")
        analyzer.save_results()
        
        print("\nAnalysis completed successfully!")
    else:
        print("No experiment results found. Please run the training experiments first.")

if __name__ == "__main__":
    main() 