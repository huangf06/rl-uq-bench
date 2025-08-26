#!/bin/bash
#SBATCH --job-name=qrdqn-uncertainty-noise0.050
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/qrdqn_uncertainty_noise0.050_%a_%j.out

# QR-DQN 不确定性退化分析 - 中等噪声组 (σ=0.050)
# 中等噪声环境，观察明显退化趋势

module load 2023
module load CUDA/12.1.1

source ~/.bashrc
conda activate zoo

# 设置种子数组
SEEDS=(101 307 911 1747 2029 2861 3253 4099 7919 9011)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

export PYTHONHASHSEED=${SEED}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "==========================================="
echo "QR-DQN UNCERTAINTY DEGRADATION ANALYSIS"
echo "==========================================="
echo "Environment: LunarLander-v3 with Noise σ=0.050 (Medium Noise)"
echo "Algorithm: QR-DQN"
echo "Seed: ${SEED}"
echo "Config: experiments/uncertainty_degradation/configs/noise0.050/qrdqn.yml"
echo ""
echo "Experiment Info:"
echo "  - Noise Level: σ=0.050 (Medium noise, clear degradation)"
echo "  - Training Steps: 800,000"
echo "  - Learning Rate: 1.5e-4"
echo "  - Buffer Size: 300,000"
echo "  - Network: [512, 512]"
echo ""
echo "Expected Performance:"
echo "  - Clear performance degradation from baseline"
echo "  - Moderate degradation trend observation"
echo ""
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "Expected duration: ~45-60 minutes"
echo "==========================================="

# 切换到项目根目录
cd /home/fhuang/rl-baselines3-zoo

# 训练命令
python train.py \
  --algo qrdqn \
  --env LunarLander-v3 \
  --conf-file experiments/uncertainty_degradation/configs/noise0.050/qrdqn.yml \
  --device cuda \
  --seed ${SEED} \
  --save-freq 50000 \
  --verbose 1

echo "==========================================="
echo "QR-DQN UNCERTAINTY DEGRADATION COMPLETED"
echo "==========================================="
echo "Completed at: $(date)"
echo ""
echo "Next Steps:"
echo "1. Run performance analysis:"
echo "   cd experiments/uncertainty_degradation/analysis"
echo "   python performance_degradation.py"
echo ""
echo "2. Compare with other noise levels"
echo "===========================================" 