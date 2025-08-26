# QR-DQN 不确定性退化分析实验总结

## 🎯 实验目标
评估 QR-DQN 算法在不同观测高斯噪声（σ）水平下的学习能力与不确定性估计性能，寻找其退化趋势和鲁棒性边界。

## 📋 实验设计

### 环境设置
- **强化学习环境**: LunarLander-v3
- **噪声类型**: 观测状态中添加高斯噪声
- **噪声标准差扫描范围**: σ ∈ {0.000, 0.025, 0.050, 0.075, 0.100}
- **随机种子**: 每个 σ 运行 10 个不同随机种子 (101, 307, 911, 1747, 2029, 2861, 3253, 4099, 7919, 9011)

### 统一 QR-DQN 配置
基于V3阶段稳定配置，重点参数包括：
- **训练步数**: 800,000
- **学习率**: 1.5e-4
- **buffer 大小**: 300,000
- **batch 大小**: 256
- **网络结构**: [512, 512]
- **其他参数**: 按稳定性经验默认设定

## 📁 文件结构
```
experiments/uncertainty_degradation/
├── README.md                           # 实验说明文档
├── EXPERIMENT_SUMMARY.md               # 本文件 - 实验总结
├── configs/                            # 配置文件目录
│   ├── noise0.000/qrdqn.yml           # 无噪声配置
│   ├── noise0.025/qrdqn.yml           # 低噪声配置
│   ├── noise0.050/qrdqn.yml           # 中等噪声配置
│   ├── noise0.075/qrdqn.yml           # 高噪声配置
│   └── noise0.100/qrdqn.yml           # 最高噪声配置
├── slurm/                              # Slurm脚本目录
│   ├── train_noise0.000.slurm         # 无噪声训练脚本
│   ├── train_noise0.025.slurm         # 低噪声训练脚本
│   ├── train_noise0.050.slurm         # 中等噪声训练脚本
│   ├── train_noise0.075.slurm         # 高噪声训练脚本
│   └── train_noise0.100.slurm         # 最高噪声训练脚本
├── analysis/                           # 分析脚本目录
│   └── performance_degradation.py      # 性能退化分析脚本
├── submit_all_experiments.sh           # 批量提交脚本
└── results/                            # 结果输出目录 (自动创建)
    ├── detailed_results.csv            # 详细结果
    ├── performance_summary.csv         # 性能汇总
    ├── performance_summary.json        # JSON格式汇总
    └── plots/                          # 可视化图表
        └── performance_degradation_trends.png
```

## 🚀 使用方法

### 1. 提交所有实验
```bash
cd experiments/uncertainty_degradation
chmod +x submit_all_experiments.sh
./submit_all_experiments.sh
```

### 2. 单独提交某个噪声水平
```bash
# 例如提交 σ=0.050 的实验
sbatch slurm/train_noise0.050.slurm
```

### 3. 分析实验结果
```bash
cd experiments/uncertainty_degradation/analysis
python performance_degradation.py
```

### 4. 查看任务状态
```bash
squeue -u $USER
```

## 📊 预期输出

### 每个 σ 对应的训练结果包括：
1. **性能指标**:
   - 平均分数、标准差
   - 最高/最低分数
   - 成功率（分数≥200的run比例）
   - 可用率（分数≥150的run比例）

2. **不确定性估计指标**:
   - CRPS (Continuous Ranked Probability Score)
   - WIS (Weighted Interval Score)
   - ACE (Average Calibration Error)

3. **可视化**:
   - 训练曲线对比
   - 不确定性退化趋势图
   - 性能vs噪声水平关系图

## 🔍 实验预期发现

### 性能退化趋势
- **σ=0.000**: 基准性能，无噪声影响
- **σ=0.025**: 轻微性能下降，观察初始退化
- **σ=0.050**: 明显性能下降，中等退化
- **σ=0.075**: 严重性能下降，高退化
- **σ=0.100**: 极限性能下降，与之前实验对比

### 关键分析点
1. **退化曲线形状**: 线性vs非线性退化
2. **鲁棒性边界**: 算法开始失效的噪声水平
3. **一致性变化**: 不同噪声水平下的方差变化
4. **不确定性质量**: UQ指标随噪声的变化

## 📈 与历史实验对比

### 与V3.1实验对比
- **相同点**: 都使用σ=0.100，可对比结果
- **不同点**: 
  - V3.1: 12e5步，学习率1.5e-4，buffer 500k
  - 本实验: 8e5步，学习率1.5e-4，buffer 300k
- **对比价值**: 验证参数选择的影响

### 与Clean环境对比
- **基准**: σ=0.000 vs 无噪声wrapper
- **性能上限**: 确定算法在理想条件下的表现

## ⚠️ 注意事项

1. **资源需求**: 每个任务需要1 GPU，预计2小时完成
2. **存储空间**: 50个训练任务，注意磁盘空间
3. **队列限制**: 建议分批提交，避免资源冲突
4. **结果验证**: 确保所有evaluations.npz文件正确生成

## 🔄 后续步骤

1. **结果分析**: 运行性能退化分析脚本
2. **可视化**: 生成退化趋势图表
3. **报告撰写**: 总结关键发现和结论
4. **方法改进**: 基于发现提出改进建议
5. **扩展实验**: 考虑其他算法或环境

## 📞 技术支持

如遇到问题，请检查：
1. Slurm脚本格式是否正确
2. 配置文件路径是否存在
3. 环境变量是否正确设置
4. 日志文件中的错误信息 