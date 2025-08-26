"""
QR-Bootstrap Hybrid DQN — 简化版本，遵循现有BootstrappedDQN模式
QR-Bootstrap Hybrid DQN — Simplified version following existing BootstrappedDQN pattern
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor

class QRBootstrapNetwork(nn.Module):
    """简化的QR-Bootstrap网络"""
    
    def __init__(self, features_dim, n_actions, n_quantiles=51, n_heads=5):
        super().__init__()
        
        self.features_dim = features_dim
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.n_heads = n_heads
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # QR分支
        self.qr_head = nn.Linear(256, n_actions * n_quantiles)
        
        # Bootstrap分支
        self.bootstrap_heads = nn.ModuleList([
            nn.Linear(256, n_actions) for _ in range(n_heads)
        ])
        
        # 简单的融合权重 (固定50-50，避免复杂性)
        self.qr_weight = 0.6
        self.bootstrap_weight = 0.4
        
        # 分位数级别
        self.register_buffer('quantile_levels', 
                           torch.linspace(0.01, 0.99, n_quantiles))
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        # QR分支
        qr_out = self.qr_head(shared_features)
        qr_quantiles = qr_out.view(-1, self.n_actions, self.n_quantiles)
        qr_q = torch.mean(qr_quantiles, dim=-1)  # [batch, n_actions]
        
        # Bootstrap分支
        bootstrap_outs = []
        for head in self.bootstrap_heads:
            bootstrap_outs.append(head(shared_features))
        bootstrap_ensemble = torch.stack(bootstrap_outs, dim=-1)  # [batch, n_actions, n_heads]
        bootstrap_q = torch.mean(bootstrap_ensemble, dim=-1)  # [batch, n_actions]
        
        # 简单融合
        fused_q = self.qr_weight * qr_q + self.bootstrap_weight * bootstrap_q
        
        return fused_q
    
    def get_qr_quantiles(self, x):
        """获取QR分位数用于UQ分析"""
        shared_features = self.shared(x)
        qr_out = self.qr_head(shared_features)
        return qr_out.view(-1, self.n_actions, self.n_quantiles)
    
    def get_bootstrap_ensemble(self, x):
        """获取Bootstrap ensemble用于UQ分析"""
        shared_features = self.shared(x)
        bootstrap_outs = []
        for head in self.bootstrap_heads:
            bootstrap_outs.append(head(shared_features))
        return torch.stack(bootstrap_outs, dim=-1)
    
    def set_training_mode(self, mode: bool) -> None:
        """Set training mode for compatibility with SB3"""
        self.train(mode)
    
    def _predict(self, obs: torch.Tensor, deterministic=None) -> torch.Tensor:
        """Predict action for compatibility with SB3 DQN"""
        q_values = self(obs)
        return q_values.argmax(dim=1)

class QRBootstrapPolicy(DQNPolicy):
    """QR-Bootstrap策略，遵循DQNPolicy模式"""
    
    def __init__(self, observation_space, action_space, lr_schedule, 
                 n_quantiles=51, n_heads=5, **kwargs):
        self.n_quantiles = n_quantiles
        self.n_heads = n_heads
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    
    def make_q_net(self):
        """创建Q网络"""
        self.features_extractor = FlattenExtractor(self.observation_space)
        return QRBootstrapNetwork(
            self.features_extractor.features_dim,
            self.action_space.n,
            n_quantiles=self.n_quantiles,
            n_heads=self.n_heads
        )

class QRBootstrapDQN(DQN):
    """QR-Bootstrap DQN，遵循现有DQN模式"""
    
    policy_aliases = {"MlpPolicy": QRBootstrapPolicy, "CnnPolicy": QRBootstrapPolicy}
    
    def __init__(self, policy, env, n_quantiles=51, n_heads=5, **kwargs):
        self.n_quantiles = n_quantiles
        self.n_heads = n_heads
        
        # 设置policy参数
        kwargs.setdefault("policy_kwargs", {})["n_quantiles"] = n_quantiles
        kwargs.setdefault("policy_kwargs", {})["n_heads"] = n_heads
        
        super().__init__(policy, env, **kwargs)
        
        # 注册分位数级别
        self.register_buffer('quantile_levels', 
                           torch.linspace(0.01, 0.99, n_quantiles))
    
    # 使用默认DQN训练方法，简化实现避免复杂性
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """兼容性方法"""
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        self._buffers[name] = tensor.to(self.device)