# QR-DQN 不确定性退化分析实验

## 🎯 项目目标
评估 QR-DQN 算法在不同观测高斯噪声（σ）水平下的学习能力与不确定性估计性能，寻找其退化趋势和鲁棒性边界。

## 📌 实验设置

### 强化学习环境
- **环境**: LunarLander-v3
- **噪声类型**: 观测状态中添加高斯噪声
- **噪声标准差扫描范围**: σ ∈ {0.000, 0.025, 0.050, 0.075, 0.100}
- **随机种子**: 每个 σ 运行 10 个不同随机种子

### 统一 QR-DQN 配置
基于V3阶段稳定配置，重点参数包括：
- **训练步数**: 800,000
- **学习率**: 1.5e-4
- **buffer 大小**: 300,000
- **batch 大小**: 256
- **网络结构**: [512, 512]
- **其他参数**: 按稳定性经验默认设定

## 📊 实验输出要求

### 每个 σ 对应的训练结果包括：
1. **性能指标**:
   - 平均分数、标准差
   - 最高/最低分数
   - 成功率（分数≥200的run比例）

2. **不确定性估计指标**:
   - CRPS (Continuous Ranked Probability Score)
   - WIS (Weighted Interval Score)
   - ACE (Average Calibration Error)

3. **可视化**:
   - 训练曲线对比
   - 不确定性退化趋势图
   - 性能vs噪声水平关系图

## 📁 目录结构
```
experiments/uncertainty_degradation/
├── configs/
│   ├── noise0.000/
│   ├── noise0.025/
│   ├── noise0.050/
│   ├── noise0.075/
│   └── noise0.100/
├── slurm/
│   ├── train_noise0.000.slurm
│   ├── train_noise0.025.slurm
│   ├── train_noise0.050.slurm
│   ├── train_noise0.075.slurm
│   └── train_noise0.100.slurm
├── analysis/
│   ├── performance_degradation.py
│   ├── uncertainty_metrics.py
│   └── visualization.py
└── results/
    ├── performance_summary.csv
    ├── uncertainty_metrics.csv
    └── plots/
```

## 🔄 实验流程
1. **配置生成**: 为每个噪声水平创建训练配置
2. **批量训练**: 使用Slurm提交训练任务
3. **结果收集**: 汇总所有种子的训练和评估结果
4. **性能分析**: 计算退化趋势和统计指标
5. **不确定性评估**: 计算UQ指标和校准性能
6. **可视化**: 生成退化趋势图和对比分析

## 📈 预期发现
- QR-DQN在不同噪声水平下的性能退化曲线
- 不确定性估计质量随噪声增加的退化模式
- 算法的鲁棒性边界和失效点
- 与其他DQN变体的对比分析基础 