#!/usr/bin/env python3
"""
Installation Verification Script for RL-UQ-Bench
Run this script after installation to verify everything is working correctly.
"""

import sys
import importlib
from pathlib import Path

def check_import(module_name, description=""):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description} (Error: {e})")
        return False

def check_file_exists(file_path, description=""):
    """Check if a file exists."""
    if Path(file_path).exists():
        print(f"✅ {file_path} - {description}")
        return True
    else:
        print(f"❌ {file_path} - {description} (File not found)")
        return False

def main():
    print("🔍 RL-UQ-Bench Installation Verification")
    print("=" * 50)
    
    success_count = 0
    total_checks = 0
    
    # Core dependencies
    print("\n📦 Core Dependencies:")
    checks = [
        ("stable_baselines3", "Stable Baselines3 RL library"),
        ("sb3_contrib", "SB3 Contrib algorithms"),
        ("gymnasium", "Gymnasium environment interface"),
        ("torch", "PyTorch deep learning framework"),
        ("numpy", "NumPy numerical computing"),
        ("pandas", "Pandas data manipulation"),
        ("matplotlib", "Matplotlib plotting"),
        ("seaborn", "Seaborn statistical visualization"),
        ("yaml", "YAML configuration parsing"),
        ("tqdm", "Progress bars"),
    ]
    
    for module, desc in checks:
        if check_import(module, desc):
            success_count += 1
        total_checks += 1
    
    # RL-UQ-Bench modules
    print("\n🎯 RL-UQ-Bench Modules:")
    uq_checks = [
        ("rl_zoo3", "Core RL Zoo3 framework"),
        ("uq_pipeline", "UQ evaluation pipeline"),
        ("rl_zoo3.qr_bootstrap_dqn", "QR-Bootstrap DQN implementation"),
        ("rl_zoo3.bootstrapped_dqn", "Bootstrapped DQN implementation"),
        ("rl_zoo3.mcdropout_dqn", "MC Dropout DQN implementation"),
    ]
    
    for module, desc in uq_checks:
        if check_import(module, desc):
            success_count += 1
        total_checks += 1
    
    # Configuration files
    print("\n⚙️  Configuration Files:")
    config_checks = [
        ("hyperparams/dqn.yml", "DQN hyperparameters"),
        ("hyperparams/qrdqn.yml", "QR-DQN hyperparameters"),
        ("hyperparams/bootstrapped_dqn.yml", "Bootstrapped DQN hyperparameters"),
        ("hyperparams/mcdropout_dqn.yml", "MC Dropout DQN hyperparameters"),
        ("uq_pipeline/configs/complete_multi_method_experiment.yml", "Multi-method experiment config"),
    ]
    
    for file_path, desc in config_checks:
        if check_file_exists(file_path, desc):
            success_count += 1
        total_checks += 1
    
    # Test basic functionality
    print("\n🧪 Basic Functionality Tests:")
    try:
        from rl_zoo3.algos import ALGOS
        if "qrdqn" in ALGOS:
            print("✅ QR-DQN algorithm registered")
            success_count += 1
        else:
            print("❌ QR-DQN algorithm not found in registry")
        total_checks += 1
        
        from uq_pipeline.utils.config_manager import load_config
        print("✅ UQ Pipeline configuration loader working")
        success_count += 1
        total_checks += 1
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        total_checks += 2
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Installation Verification Summary:")
    print(f"✅ Passed: {success_count}/{total_checks} checks")
    
    if success_count == total_checks:
        print("🎉 Perfect! RL-UQ-Bench is ready to use!")
        print("\n🚀 Quick Start:")
        print("   python train.py --algo qrdqn --env LunarLander-v3")
        print("   python -m uq_pipeline.runner --config uq_pipeline/configs/complete_multi_method_experiment.yml")
        return 0
    else:
        print(f"⚠️  {total_checks - success_count} issues found. Please check installation.")
        print("\n🔧 Installation commands:")
        print("   pip install -e .")
        print("   # OR")  
        print("   pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())