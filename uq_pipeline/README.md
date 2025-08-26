# UQ Pipeline: 不确定性量化评估系统

一个模块化、可扩展的强化学习不确定性量化方法评估管道。

## 📁 目录结构

```
uq_pipeline/                        ← 📦 项目主目录：只放代码
├── configs/
│   └── experiment_lunarlander.yml   ← 实验配置文件
├── stages/                          ← 管道阶段模块
│   ├── stage0_config.py            ← 配置加载与验证
│   ├── stage1_dataset_builder.py   ← 数据集构建
│   ├── stage2_performance.py       ← 性能评估
│   ├── stage3_q_extractor.py       ← Q值分布提取
│   ├── stage4_metrics.py           ← 原始指标计算
│   ├── stage5_calibration.py       ← 校准与调整指标
│   └── stage6_report.py            ← 报告生成
├── utils/                           ← 工具模块
│   ├── context.py                  ← 实验上下文管理
│   ├── path_manager.py             ← 路径管理
│   ├── logging_utils.py            ← 日志工具
│   └── data_format.py              ← 数据格式工具
└── runner.py                        ← 主运行器

uq_results/                          ← 📂 只保存数据和实验输出
├── data/                           ← 清洁数据集
│   └── LunarLander-v3/
│       └── uncertainty_degradation_noise0.050/
│           └── eval_dataset.xz
└── results/                        ← 实验结果
    └── LunarLander-v3/
        └── uncertainty_degradation_noise0.050/
            └── qrdqn/
                └── seed_101/
                    ├── performance.json
                    ├── q_values.xz
                    ├── metrics_raw.csv
                    ├── calibration_params.json
                    ├── metrics_calibrated.csv
                    └── summary.json
```

## 🚀 快速开始

### 1. 基本使用

```bash
# 运行完整管道
python -m uq_pipeline.runner --config uq_pipeline/configs/experiment_lunarlander.yml

# 详细日志
python -m uq_pipeline.runner --config uq_pipeline/configs/experiment_lunarlander.yml --verbose

# 干运行（验证配置）
python -m uq_pipeline.runner --config uq_pipeline/configs/experiment_lunarlander.yml --dry-run
```

### 2. 阶段控制

```bash
# 运行特定阶段
python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml --stages 1,2,3

# 从特定阶段开始运行
python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml --from-stage 3

# 只运行单个阶段
python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml --only-stage 4
```

### 3. 恢复和强制模式

```bash
# 恢复中断的运行（跳过已完成的实验）
python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml --resume

# 强制重新运行所有实验
python -m uq_pipeline.runner --config configs/experiment_lunarlander.yml --force
```

## ⚙️ 配置文件

`configs/experiment_lunarlander.yml`:

```yaml
env_id: LunarLander-v3
uq_methods: [qrdqn, bootstrapped_dqn, mcdropout_dqn, dqn]
env_types:
  - uncertainty_degradation_noise0.000
  - uncertainty_degradation_noise0.025
  - uncertainty_degradation_noise0.050
  - uncertainty_degradation_noise0.075
  - uncertainty_degradation_noise0.100
seeds: [101, 202, 303]
data_root: uq_results/data/
results_root: uq_results/results/
eval_episodes: 50
```

## 🧩 管道阶段

### Stage 0: 配置验证
- 验证配置文件格式和内容
- 检查路径和目录可访问性
- 验证UQ方法支持

### Stage 1: 数据集构建
- 为每个环境类型生成清洁评估数据集
- 支持不同噪声级别的环境配置
- 数据压缩存储

### Stage 2: 性能评估
- 评估训练模型在清洁数据集上的性能
- 计算回合奖励、成功率等指标
- 测量推理时间

### Stage 3: Q值提取
- 从UQ模型提取Q值分布
- 支持QR-DQN、Bootstrapped DQN、MC Dropout
- 压缩存储Q值数组

### Stage 4: 原始指标计算
- 计算全面的不确定性量化指标
- 分布指标、动作选择指标、置信度指标
- 方法特定指标

### Stage 5: 校准与调整
- 执行不确定性校准
- 温度缩放、Platt缩放、等温回归
- 计算校准后指标

### Stage 6: 报告生成
- 汇总所有实验结果
- 生成对比表格和可视化
- 创建综合HTML/PDF报告

## 📊 输出文件

每个实验组合会生成以下文件：

- `performance.json`: 性能指标
- `q_values.xz`: Q值分布（压缩）
- `metrics_raw.csv`: 原始UQ指标
- `calibration_params.json`: 校准参数
- `metrics_calibrated.csv`: 校准后指标
- `summary.json`: 实验总结和完成标记

## 🔧 扩展和自定义

### 添加新的UQ方法

1. 在配置文件中添加方法名
2. 在 `stage3_q_extractor.py` 中实现方法特定的Q值提取
3. 在 `stage4_metrics.py` 中添加方法特定指标

### 添加新的校准方法

1. 在 `stage5_calibration.py` 中实现新的校准函数
2. 更新 `get_supported_calibration_methods()`
3. 在校准管道中集成新方法

### 自定义报告

1. 修改 `stage6_report.py` 中的报告生成函数
2. 添加新的可视化或分析表格
3. 自定义HTML模板

## 📋 依赖要求

- Python 3.8+
- pandas
- numpy
- PyYAML
- pathlib
- logging
- lzma (用于压缩)

## 🐛 故障排除

### 常见问题

1. **配置验证失败**
   - 检查YAML语法
   - 验证路径是否存在
   - 确认UQ方法名称正确

2. **内存不足**
   - 减少eval_episodes数量
   - 使用更少的种子
   - 启用压缩存储

3. **磁盘空间不足**
   - 检查估计的存储需求
   - 清理之前的结果
   - 使用外部存储

### 日志和调试

```bash
# 启用详细日志
python -m uq_pipeline.runner --config configs/experiment.yml --verbose

# 保存日志到文件
python -m uq_pipeline.runner --config configs/experiment.yml --log-file pipeline.log

# 检查特定阶段
python -m uq_pipeline.runner --config configs/experiment.yml --only-stage 2 --verbose
```

## 📈 性能优化

- 使用 `--parallel` 启用并行处理
- 调整 `--max-workers` 控制并发数
- 使用 `--resume` 避免重复计算
- 定期清理临时文件

## 📝 开发状态

当前代码提供了完整的框架结构和接口，所有函数都有详细的文档字符串和TODO标记。需要根据具体的UQ方法和环境实现实际的逻辑。

核心特性：
- ✅ 模块化设计
- ✅ 配置管理
- ✅ 路径管理
- ✅ 日志系统
- ✅ 数据格式工具
- ✅ 阶段化执行
- ✅ 命令行界面
- 🔄 实现逻辑（TODO）

下一步：根据具体需求实现各个模块的核心逻辑。