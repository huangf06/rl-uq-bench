#!/bin/bash
# 批量训练多种UQ方法：Bootstrapped DQN, MC Dropout DQN, DQN Baseline
# 5个噪声级别 × 3种方法 × 10个种子 = 150个训练任务

set -e

# 配置参数
NOISE_LEVELS=(0.000 0.025 0.050 0.075 0.100)
METHODS=(bootstrapped_dqn mcdropout_dqn dqn)
SEEDS=(101 307 911 1747 2029 2861 3253 4099 7919 9011)

# 训练参数
ENV_ID="LunarLander-v3"
BASE_LOG_DIR="logs/multi_env_experiments"

echo "=== UQ Multi-Method Training Pipeline ==="
echo "Methods: ${METHODS[@]}"
echo "Noise levels: ${NOISE_LEVELS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Total jobs: $((${#METHODS[@]} * ${#NOISE_LEVELS[@]} * ${#SEEDS[@]}))"
echo

# 函数：训练单个模型
train_model() {
    local method=$1
    local noise=$2
    local seed=$3
    
    local config_path="experiments/uncertainty_degradation/configs/noise${noise}/${method}.yml"
    local log_path="${BASE_LOG_DIR}/${ENV_ID}/uncertainty_degradation_noise${noise}/${method}/seed_${seed}_1"
    
    echo "Training: ${method} | noise=${noise} | seed=${seed}"
    
    # 创建日志目录
    mkdir -p "$(dirname "${log_path}")"
    
    # 启动训练
    python train.py \
        --algo "${method}" \
        --env "${ENV_ID}" \
        --conf-file "${config_path}" \
        --seed "${seed}" \
        --log-folder "${log_path}" \
        --verbose 1 \
        --tensorboard-log "${log_path}/tensorboard" \
        2>&1 | tee "${log_path}/training.log"
        
    echo "Completed: ${method} | noise=${noise} | seed=${seed}"
}

# 函数：批量训练（串行）
train_sequential() {
    local job_count=0
    local total_jobs=$((${#METHODS[@]} * ${#NOISE_LEVELS[@]} * ${#SEEDS[@]}))
    
    for noise in "${NOISE_LEVELS[@]}"; do
        for method in "${METHODS[@]}"; do
            # 跳过已存在的QRDQN模型
            if [ "$method" = "qrdqn" ]; then
                echo "Skipping existing QRDQN models for noise=${noise}"
                continue
            fi
            
            for seed in "${SEEDS[@]}"; do
                job_count=$((job_count + 1))
                echo "Progress: ${job_count}/${total_jobs}"
                
                train_model "$method" "$noise" "$seed"
            done
        done
    done
}

# 函数：并行训练（SLURM）
train_slurm() {
    for noise in "${NOISE_LEVELS[@]}"; do
        for method in "${METHODS[@]}"; do
            # 跳过QRDQN
            if [ "$method" = "qrdqn" ]; then
                continue
            fi
            
            for seed in "${SEEDS[@]}"; do
                local job_name="${method}_${noise}_${seed}"
                local config_path="experiments/uncertainty_degradation/configs/noise${noise}/${method}.yml"
                local log_path="${BASE_LOG_DIR}/${ENV_ID}/uncertainty_degradation_noise${noise}/${method}/seed_${seed}_1"
                
                echo "Submitting SLURM job: ${job_name}"
                
                sbatch --job-name="${job_name}" \
                       --partition=gpu \
                       --time=04:00:00 \
                       --mem=8G \
                       --gres=gpu:1 \
                       --wrap="python train.py --algo ${method} --env ${ENV_ID} --conf-file ${config_path} --seed ${seed} --log-folder ${log_path} --verbose 1"
            done
        done
    done
}

# 函数：检查训练进度
check_progress() {
    echo "=== Training Progress Check ==="
    
    for noise in "${NOISE_LEVELS[@]}"; do
        for method in "${METHODS[@]}"; do
            if [ "$method" = "qrdqn" ]; then
                continue
            fi
            
            local completed_count=0
            for seed in "${SEEDS[@]}"; do
                local model_path="${BASE_LOG_DIR}/${ENV_ID}/uncertainty_degradation_noise${noise}/${method}/seed_${seed}_1/best_model.zip"
                if [ -f "$model_path" ]; then
                    completed_count=$((completed_count + 1))
                fi
            done
            
            echo "${method} | noise=${noise}: ${completed_count}/${#SEEDS[@]} completed"
        done
    done
}

# 主执行逻辑
case "${1:-sequential}" in
    "sequential")
        echo "Starting sequential training..."
        train_sequential
        ;;
    "slurm")
        echo "Submitting SLURM jobs..."
        train_slurm
        ;;
    "check")
        check_progress
        ;;
    *)
        echo "Usage: $0 [sequential|slurm|check]"
        echo "  sequential: Train models one by one (default)"
        echo "  slurm:      Submit parallel SLURM jobs"
        echo "  check:      Check training progress"
        exit 1
        ;;
esac

echo "Script completed!"