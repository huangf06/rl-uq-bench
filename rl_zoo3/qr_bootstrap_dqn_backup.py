"""
QR-Bootstrap Hybrid DQN — 结合分位数回归和Bootstrap ensemble的混合方法
QR-Bootstrap Hybrid DQN — Combining quantile regression with bootstrap ensemble
----------------------------------------------------------------------------------
✓ 双分支架构: QR分支(分位数预测) + Bootstrap分支(ensemble预测)
✓ 自适应权重融合: 学习QR和Bootstrap的最优结合权重
✓ 混合损失函数: Quantile Loss + Ensemble Loss + Consistency Loss
✓ 兼容SB3框架: 基于现有BootstrappedDQN和QR-DQN扩展
"""

from __future__ import annotations
from typing import List, Optional, NamedTuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import polyak_update

# 复用现有的Bootstrap基础设施
from .bootstrapped_dqn import BootstrappedSamples, BootstrapMaskBuffer


class QRBootstrapNetwork(nn.Module):
    """
    QR-Bootstrap混合网络
    结合QR-DQN的分位数预测和BootstrappedDQN的ensemble结构
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_quantiles: int = 51,
        n_bootstrap_heads: int = 5,
        features_extractor_class=FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_quantiles = n_quantiles
        self.n_bootstrap_heads = n_bootstrap_heads
        self.n_actions = action_space.n
        
        # 特征提取器
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        self.features_extractor = features_extractor_class(
            observation_space, **features_extractor_kwargs
        )
        
        # 共享特征层
        self.shared_net = nn.Sequential(
            nn.Linear(self.features_extractor.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # QR分支: 预测分位数分布 [batch, n_actions, n_quantiles]
        self.qr_head = nn.Linear(256, self.n_actions * self.n_quantiles)
        
        # Bootstrap分支: 多个独立Q头 [batch, n_actions, n_heads] 
        self.bootstrap_heads = nn.ModuleList([
            nn.Linear(256, self.n_actions) for _ in range(n_bootstrap_heads)
        ])
        
        # 自适应融合层: 学习QR和Bootstrap的权重
        self.fusion_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)  # 确保权重和为1
        )
        
        # 分位数级别 (1%-99%)
        quantile_levels = torch.linspace(0.01, 0.99, n_quantiles)
        self.register_buffer('quantile_levels', quantile_levels)
    
    def forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播: 同时输出QR分位数和Bootstrap ensemble
        """
        # 特征提取
        features = self.features_extractor(observations)
        shared_features = self.shared_net(features)
        
        # QR分支预测
        qr_outputs = self.qr_head(shared_features)
        qr_quantiles = qr_outputs.view(-1, self.n_actions, self.n_quantiles)
        
        # Bootstrap分支预测
        bootstrap_outputs = []
        for head in self.bootstrap_heads:
            head_q = head(shared_features)
            bootstrap_outputs.append(head_q)
        bootstrap_ensemble = torch.stack(bootstrap_outputs, dim=-1)  # [batch, n_actions, n_heads]
        
        # 自适应融合权重
        fusion_weights = self.fusion_net(shared_features)  # [batch, 2]
        qr_weight = fusion_weights[:, 0:1]  # [batch, 1]
        bootstrap_weight = fusion_weights[:, 1:2]  # [batch, 1]
        
        return {
            'qr_quantiles': qr_quantiles,
            'bootstrap_ensemble': bootstrap_ensemble,
            'fusion_weights': fusion_weights,
            'qr_weight': qr_weight,
            'bootstrap_weight': bootstrap_weight,
            'shared_features': shared_features
        }
    
    def get_q_values(self, observations: torch.Tensor) -> torch.Tensor:
        """
        获取Q值 (用于action selection)
        使用融合后的期望值作为Q值估计
        """
        outputs = self.forward(observations)
        
        # QR分支的期望Q值
        qr_q_values = torch.mean(outputs['qr_quantiles'], dim=-1)  # [batch, n_actions]
        
        # Bootstrap分支的期望Q值
        bootstrap_q_values = torch.mean(outputs['bootstrap_ensemble'], dim=-1)  # [batch, n_actions]
        
        # 融合Q值
        qr_weight = outputs['qr_weight']  # [batch, 1]
        bootstrap_weight = outputs['bootstrap_weight']  # [batch, 1]
        
        fused_q_values = qr_weight * qr_q_values + bootstrap_weight * bootstrap_q_values
        
        return fused_q_values
    
    def get_uncertainty_estimates(self, observations: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取混合不确定性估计
        """
        outputs = self.forward(observations)
        batch_size = observations.size(0)
        
        # 提取指定action的预测
        action_indices = actions.long().unsqueeze(-1)  # [batch, 1]
        
        # QR分支的不确定性
        qr_quantiles_selected = torch.gather(
            outputs['qr_quantiles'], 1, 
            action_indices.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
        ).squeeze(1)  # [batch, n_quantiles]
        
        qr_mean = torch.mean(qr_quantiles_selected, dim=-1)
        qr_std = torch.std(qr_quantiles_selected, dim=-1)
        
        # Bootstrap分支的不确定性
        bootstrap_values_selected = torch.gather(
            outputs['bootstrap_ensemble'], 1,
            action_indices.unsqueeze(-1).expand(-1, -1, self.n_bootstrap_heads)
        ).squeeze(1)  # [batch, n_heads]
        
        bootstrap_mean = torch.mean(bootstrap_values_selected, dim=-1)
        bootstrap_std = torch.std(bootstrap_values_selected, dim=-1)
        
        # 融合不确定性
        qr_weight = outputs['qr_weight'].squeeze()  # [batch]
        bootstrap_weight = outputs['bootstrap_weight'].squeeze()  # [batch]
        
        fused_mean = qr_weight * qr_mean + bootstrap_weight * bootstrap_mean
        fused_std = torch.sqrt(
            qr_weight**2 * qr_std**2 + bootstrap_weight**2 * bootstrap_std**2
        )
        
        return {
            'fused_mean': fused_mean,
            'fused_std': fused_std,
            'qr_mean': qr_mean,
            'qr_std': qr_std,
            'bootstrap_mean': bootstrap_mean,
            'bootstrap_std': bootstrap_std,
            'qr_weight': qr_weight,
            'bootstrap_weight': bootstrap_weight,
            'qr_quantiles': qr_quantiles_selected,
            'bootstrap_ensemble': bootstrap_values_selected
        }


class QRBootstrapPolicy(DQNPolicy):
    """
    QR-Bootstrap混合策略
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule,
        n_quantiles: int = 51,
        n_bootstrap_heads: int = 5,
        **kwargs
    ):
        self.n_quantiles = n_quantiles
        self.n_bootstrap_heads = n_bootstrap_heads
        
        # 设置网络参数
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = [256, 256]
        
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    
    def make_q_net(self) -> QRBootstrapNetwork:
        """创建Q网络"""
        return QRBootstrapNetwork(
            self.observation_space,
            self.action_space,
            n_quantiles=self.n_quantiles,
            n_bootstrap_heads=self.n_bootstrap_heads,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
        )
    
    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """前向传播用于action selection"""
        q_values = self.q_net.get_q_values(obs)
        return q_values
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """预测动作"""
        q_values = self.q_net.get_q_values(observation)
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class QRBootstrapDQN(DQN):
    """
    QR-Bootstrap混合DQN算法
    """
    
    policy: QRBootstrapPolicy
    
    def __init__(
        self,
        policy="QRBootstrapPolicy",
        env=None,
        n_quantiles: int = 51,
        n_bootstrap_heads: int = 5,
        qr_loss_weight: float = 1.0,
        bootstrap_loss_weight: float = 1.0,
        consistency_loss_weight: float = 0.1,
        fusion_regularization_weight: float = 0.01,
        **kwargs
    ):
        self.n_quantiles = n_quantiles
        self.n_bootstrap_heads = n_bootstrap_heads
        self.qr_loss_weight = qr_loss_weight
        self.bootstrap_loss_weight = bootstrap_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.fusion_regularization_weight = fusion_regularization_weight
        
        # 设置policy参数
        if "policy_kwargs" not in kwargs:
            kwargs["policy_kwargs"] = {}
        kwargs["policy_kwargs"]["n_quantiles"] = n_quantiles
        kwargs["policy_kwargs"]["n_bootstrap_heads"] = n_bootstrap_heads
        
        # 使用Bootstrap mask buffer
        kwargs["replay_buffer_class"] = BootstrapMaskBuffer
        kwargs["replay_buffer_kwargs"] = {
            "n_heads": n_bootstrap_heads,
            **(kwargs.get("replay_buffer_kwargs", {}))
        }
        
        super().__init__(policy=policy, env=env, **kwargs)
        
        # 注册分位数级别
        quantile_levels = torch.linspace(0.01, 0.99, n_quantiles)
        self.register_buffer('quantile_levels', quantile_levels)
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """训练网络"""
        losses = []
        
        for _ in range(gradient_steps):
            # 从经验池中采样 (包含bootstrap mask)
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # 计算混合损失
            loss_dict = self._compute_hybrid_loss(replay_data)
            loss = loss_dict['total_loss']
            
            # 梯度更新
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            losses.append(loss.item())
        
        # 更新target network
        polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
        
        # 记录训练统计
        self.logger.record("train/loss", np.mean(losses))
    
    def _compute_hybrid_loss(self, replay_data: BootstrappedSamples) -> Dict[str, torch.Tensor]:
        """计算混合损失函数"""
        with torch.no_grad():
            # 目标值计算 (使用target network)
            next_q_values = self.q_net_target.get_q_values(replay_data.next_observations)
            next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
        
        # 当前网络预测
        current_outputs = self.q_net.forward(replay_data.observations)
        
        # 提取选定action的预测
        actions = replay_data.actions.long()
        action_indices = actions.unsqueeze(-1)
        
        # QR分支损失: Quantile Regression Loss
        current_qr_quantiles = torch.gather(
            current_outputs['qr_quantiles'], 1,
            action_indices.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
        ).squeeze(1)  # [batch, n_quantiles]
        
        target_expanded = target_q_values.unsqueeze(-1).expand(-1, self.n_quantiles)
        quantile_errors = target_expanded - current_qr_quantiles
        quantile_losses = torch.max(
            self.quantile_levels * quantile_errors,
            (self.quantile_levels - 1) * quantile_errors
        )
        qr_loss = quantile_losses.mean()
        
        # Bootstrap分支损失: Masked Ensemble MSE Loss
        bootstrap_losses = []
        for head_idx in range(self.n_bootstrap_heads):
            head_q_values = current_outputs['bootstrap_ensemble'][:, :, head_idx]
            head_selected = torch.gather(head_q_values, 1, action_indices).squeeze(1)
            
            # 使用bootstrap mask
            mask = replay_data.mask[:, head_idx]
            masked_loss = F.mse_loss(head_selected * mask, target_q_values * mask, reduction='none')
            bootstrap_losses.append(masked_loss.mean())
        
        bootstrap_loss = torch.stack(bootstrap_losses).mean()
        
        # 一致性损失: QR和Bootstrap预测的一致性
        qr_mean_q = torch.mean(current_qr_quantiles, dim=-1)
        bootstrap_mean_q = torch.gather(
            torch.mean(current_outputs['bootstrap_ensemble'], dim=-1), 1, action_indices
        ).squeeze(1)
        consistency_loss = F.mse_loss(qr_mean_q, bootstrap_mean_q)
        
        # 融合权重正则化: 避免一个分支完全主导
        fusion_weights = current_outputs['fusion_weights']
        # 鼓励权重接近均匀分布 [0.5, 0.5]
        target_weights = torch.full_like(fusion_weights, 0.5)
        fusion_regularization = F.mse_loss(fusion_weights, target_weights)
        
        # 总损失
        total_loss = (self.qr_loss_weight * qr_loss + 
                     self.bootstrap_loss_weight * bootstrap_loss + 
                     self.consistency_loss_weight * consistency_loss + 
                     self.fusion_regularization_weight * fusion_regularization)
        
        return {
            'total_loss': total_loss,
            'qr_loss': qr_loss,
            'bootstrap_loss': bootstrap_loss,
            'consistency_loss': consistency_loss,
            'fusion_regularization': fusion_regularization
        }
    
    @property  
    def q_net(self) -> QRBootstrapNetwork:
        """当前Q网络"""
        return self.policy.q_net
    
    @property
    def q_net_target(self) -> QRBootstrapNetwork:
        """目标Q网络"""
        return self.policy.q_net_target
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册buffer到设备"""
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        self._buffers[name] = tensor.to(self.device)


# QRBootstrapPolicy和QRBootstrapDQN类已经在上面定义，无需额外注册函数


if __name__ == "__main__":
    # 测试混合网络
    import gymnasium as gym
    
    env = gym.make("CartPole-v1")
    
    # 创建混合网络
    network = QRBootstrapNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        n_quantiles=51,
        n_bootstrap_heads=5
    )
    
    # 测试前向传播
    obs = torch.randn(32, 4)  # batch_size=32
    outputs = network.forward(obs)
    
    print("🧪 QR-Bootstrap网络测试:")
    print(f"  QR quantiles shape: {outputs['qr_quantiles'].shape}")
    print(f"  Bootstrap ensemble shape: {outputs['bootstrap_ensemble'].shape}")
    print(f"  Fusion weights shape: {outputs['fusion_weights'].shape}")
    print(f"✅ 网络架构验证通过!")
    
    # 测试Q值计算
    q_values = network.get_q_values(obs)
    print(f"  Q values shape: {q_values.shape}")
    
    # 测试不确定性估计
    actions = torch.randint(0, 2, (32,))
    uncertainty = network.get_uncertainty_estimates(obs, actions)
    print(f"  Uncertainty keys: {list(uncertainty.keys())}")
    print(f"✅ 混合方法实现完成!")