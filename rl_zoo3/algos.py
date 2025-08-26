from typing import Dict, Type

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import ARS, QRDQN, TQC, TRPO, CrossQ, RecurrentPPO

from rl_zoo3.bootstrapped_dqn import BootstrappedDQN
from rl_zoo3.mcdropout_dqn import MCDropoutDQN
from rl_zoo3.qr_bootstrap_dqn import QRBootstrapDQN

ALGOS: Dict[str, Type] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "ars": ARS,
    "crossq": CrossQ,
    "qrdqn": QRDQN,
    "tqc": TQC,
    "trpo": TRPO,
    "ppo_lstm": RecurrentPPO,
    
    "bootstrapped_dqn": BootstrappedDQN,
    "mcdropout_dqn": MCDropoutDQN,
    "qr_bootstrap_dqn": QRBootstrapDQN,
}
