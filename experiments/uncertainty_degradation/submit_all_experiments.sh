#!/bin/bash
# QR-DQN 不确定性退化分析 - 批量提交脚本
# 按顺序提交所有噪声水平的训练任务

echo "=========================================="
echo "QR-DQN 不确定性退化分析实验"
echo "批量提交所有噪声水平的训练任务"
echo "=========================================="

# 设置实验参数
SLURM_DIR="experiments/uncertainty_degradation/slurm"
NOISE_LEVELS=("0.000" "0.025" "0.050" "0.075" "0.100")

# 创建日志目录
mkdir -p logs

# 按顺序提交任务
for noise_level in "${NOISE_LEVELS[@]}"; do
    echo ""
    echo "提交噪声水平 σ=$noise_level 的训练任务..."
    
    # 提交Slurm任务
    job_id=$(sbatch $SLURM_DIR/train_noise$noise_level.slurm | awk '{print $4}')
    
    if [ $? -eq 0 ]; then
        echo "✓ 任务提交成功! Job ID: $job_id"
        echo "  噪声水平: σ=$noise_level"
        echo "  任务数量: 10个种子 (array job)"
        echo "  预计时间: 2小时"
        echo "  分区: gpu_a100"
        echo "  资源: 1 GPU, 18 CPU cores"
    else
        echo "✗ 任务提交失败!"
        exit 1
    fi
    
    # 等待一段时间再提交下一个任务，避免资源冲突
    echo "等待30秒后提交下一个任务..."
    sleep 30
done

echo ""
echo "=========================================="
echo "所有任务提交完成!"
echo ""
echo "任务状态查询命令:"
echo "  squeue -u $USER"
echo ""
echo "结果分析命令:"
echo "  cd experiments/uncertainty_degradation/analysis"
echo "  python performance_degradation.py"
echo ""
echo "实验说明:"
echo "- 5个噪声水平: σ ∈ {0.000, 0.025, 0.050, 0.075, 0.100}"
echo "- 每个水平10个种子，总共50个训练任务"
echo "- 统一配置: 800k步，学习率1.5e-4，buffer 300k"
echo "- 目标: 分析QR-DQN性能退化趋势"
echo "==========================================" 