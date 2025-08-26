"""
Bootstrapped DQN — SB3-compatible implementation
------------------------------------------------
✓ Multi-head Q-network
✓ ReplayBuffer with bootstrap masks
✓ Per-env head switching (on episode end)
✓ ε-greedy handled by SB3; _predict outputs greedy action
✓ Evaluation uses ensemble (mean over heads) for stability
"""

from __future__ import annotations
from typing import List, Optional, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize


# ───────────────────────── Bootstrap Sample ────────────────────────── #

class BootstrappedSamples(NamedTuple):
    observations:      torch.Tensor
    actions:           torch.Tensor
    next_observations: torch.Tensor
    dones:             torch.Tensor
    rewards:           torch.Tensor
    mask:              torch.Tensor          # [batch, n_heads]


# ───────────────────────── Replay Buffer ───────────────────────────── #

class BootstrapMaskBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device,
        *,
        n_envs: int = 1,
        n_heads: int = 10,
        bootstrap_prob: float = 0.5,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.n_heads = n_heads
        self.bootstrap_prob = float(bootstrap_prob)
        self.masks = np.zeros((buffer_size, n_heads), dtype=np.bool_)

    # add() identical to parent, extra mask handling
    def add(self, obs, next_obs, action, reward, done, infos):
        obs, next_obs = map(np.atleast_2d, (obs, next_obs))
        action, reward, done = map(np.atleast_2d, (action, reward, done))
        n_envs = obs.shape[0]

        new_masks = np.random.binomial(1, self.bootstrap_prob, size=(n_envs, self.n_heads))
        zero_rows = ~new_masks.any(axis=1)
        if zero_rows.any():
            new_masks[zero_rows, np.random.randint(0, self.n_heads, zero_rows.sum())] = 1

        super().add(obs, next_obs, action, reward, done, infos)
        idx = (self.pos - n_envs + np.arange(n_envs)) % self.buffer_size
        self.masks[idx] = new_masks

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> BootstrappedSamples:
        upper = self.buffer_size if self.full else self.pos
        idx = np.random.randint(0, upper, size=batch_size)
        data = self._get_samples(idx, env=env)
        masks = torch.as_tensor(self.masks[idx], device=self.device, dtype=torch.float32)
        return BootstrappedSamples(*data[:-1], masks)


# ───────────────────────── Multi-head Network ───────────────────────── #

class MultiHeadQNet(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, n_heads: int, n_envs: int = 1, net_arch: List[int] = [256, 256]):
        super().__init__()
        self.action_dim, self.n_heads = act_dim, n_heads

        layers, last = [], in_dim
        for h in net_arch:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(last, act_dim) for _ in range(n_heads)])

        self.register_buffer("_current_heads", torch.randint(0, n_heads, size=(n_envs,)))

    # ----- forward: [B, H, A]
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(obs)
        return torch.stack([head(feat) for head in self.heads], dim=1)

    # ----- inference helpers
    def switch_heads(self):
        self._current_heads.random_(0, self.n_heads)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def _predict(self, obs: torch.Tensor, *, ensemble: bool = False, deterministic=None) -> torch.Tensor:
        q = self(obs)                    # [B, H, A]
        if ensemble:                     # used in evaluation
            return q.mean(dim=1).argmax(dim=1)

        heads = self._current_heads
        if heads.numel() == 1 and q.size(0) > 1:
            heads = heads.repeat(q.size(0))
        q_h = q[torch.arange(q.size(0), device=q.device), heads]
        return q_h.argmax(dim=1)


# ───────────────────────── Policy wrapper ──────────────────────────── #

class MultiHeadQPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, n_heads=10, **kw):
        self.n_heads = n_heads
        super().__init__(observation_space, action_space, lr_schedule, **kw)

    def make_q_net(self):
        self.features_extractor = FlattenExtractor(self.observation_space)
        n_envs = getattr(self.env, "num_envs", 1) if hasattr(self, "env") else 1
        return MultiHeadQNet(self.features_extractor.features_dim, self.action_space.n, self.n_heads, n_envs)
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not self.training:
            obs_tensor, _ = self.obs_to_tensor(observation)
            actions = self.q_net._predict(obs_tensor, ensemble=True)
            return actions.cpu().numpy(), state
        return super().predict(observation, state, episode_start, deterministic)


# ───────────────────────── Head-switch Callback ────────────────────── #

class HeadSwitchCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_rollout_start(self) -> None:
        self.model.policy.q_net.switch_heads()

    def _on_step(self) -> bool:        
        if np.any(self.locals["dones"]):
            self.model.policy.q_net.switch_heads()
        return True


# ───────────────────────── Main Algorithm ──────────────────────────── #

class BootstrappedDQN(DQN):
    policy_aliases = {"MlpPolicy": MultiHeadQPolicy, "CnnPolicy": MultiHeadQPolicy}

    def __init__(self, policy, env, n_heads=20, bootstrap_prob=0.5, **kw):
        self.n_heads, self.bootstrap_prob = n_heads, bootstrap_prob
        kw.setdefault("policy_kwargs", {})["n_heads"] = n_heads
        super().__init__(policy, env, **kw)

        self.head_switch_cb = HeadSwitchCallback()
        self.callbacks = []
        # Don't call _setup_model() here - it's already called by parent

    # ----- setup
    def _setup_model(self):
        super()._setup_model()
        self.replay_buffer = BootstrapMaskBuffer(
            self.buffer_size, self.observation_space, self.action_space, self.device,
            n_envs=self.n_envs, n_heads=self.n_heads, bootstrap_prob=self.bootstrap_prob
        )
        # init heads
        init = torch.randint(0, self.n_heads, (self.n_envs,), device=self.device)
        self.policy.q_net._current_heads      = init
        self.policy.q_net_target._current_heads = init.clone()

    # ----- overwrite learn to inject callback
    def learn(self, *args, **kw):
        if self.head_switch_cb not in self.callbacks:
            self.callbacks.append(self.head_switch_cb)
        return super().learn(*args, **kw)

    # ----- core training step
    def train(self, gradient_steps: int, batch_size: int = 256):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            mask = data.mask                                      # [B, H]

            with torch.no_grad():
                next_q_online  = self.policy.q_net(data.next_observations)
                next_act       = next_q_online.argmax(dim=2, keepdim=True)
                next_q_target  = self.policy.q_net_target(data.next_observations)
                next_q         = torch.gather(next_q_target, 2, next_act).squeeze(2)
                target_q       = data.rewards + (1 - data.dones) * self.gamma * next_q

            cur_q_all = self.policy.q_net(data.observations)
            act_idx   = data.actions.view(-1, 1, 1).expand(-1, self.n_heads, 1)
            cur_q     = torch.gather(cur_q_all, 2, act_idx).squeeze(2)

            td_err = F.smooth_l1_loss(cur_q, target_q, reduction="none")
            loss   = (td_err * mask).sum() / mask.sum().clamp_min(1)

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            losses.append(loss.item())
            self._n_updates += 1

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.policy.q_net.parameters(),
                              self.policy.q_net_target.parameters(),
                              self.tau)
                self.policy.q_net_target._current_heads = self.policy.q_net._current_heads.clone()

        if losses:
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))
