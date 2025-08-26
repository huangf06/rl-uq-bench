from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import Schedule, GymEnv
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.utils import polyak_update

# ─────────────────────────── Network ──────────────────────────── #

class DropoutMLP(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, arch: List[int], p: float):
        layers: List[nn.Module] = []
        last = in_dim
        for h in arch:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p)]
            last = h
        layers.append(nn.Linear(last, out_dim))
        super().__init__(*layers)


class DropoutQNet(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        feature_extractor_class: Type[BaseFeaturesExtractor],
        feature_extractor_kwargs: Dict[str, Any],
        net_arch: List[int],
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.features_extractor = feature_extractor_class(
            observation_space, **feature_extractor_kwargs
        )
        feat_dim = self.features_extractor.features_dim
        self.mlp = DropoutMLP(feat_dim, action_dim, net_arch, dropout_p)

    def forward(self, obs: torch.Tensor, *, training: Optional[bool] = None) -> torch.Tensor:
        if obs.dtype != torch.float32:
            obs = obs.float()
        if obs.ndim in (1, 3):
            obs = obs.unsqueeze(0)
        if obs.ndim == 4:
            obs = obs / 255.0
        feats = self.features_extractor(obs)
        if training is not None and training != self.mlp.training:
            self.mlp.train(training)
        return self.mlp(feats)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(obs, training=False)
        return q_values.argmax(dim=1)

    def set_training_mode(self, mode: bool) -> None:
        super().train(mode)
        self.features_extractor.train(mode)
        self.mlp.train(mode)


# ─────────────────────────── Policy ──────────────────────────── #

class MCDropoutQPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        *,
        dropout_p: float = 0.2,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        **policy_kwargs,
    ):
        # Extract parameters from policy_kwargs if they exist
        if "net_arch" in policy_kwargs:
            net_arch = policy_kwargs.pop("net_arch")
        if "activation_fn" in policy_kwargs:
            activation_fn = policy_kwargs.pop("activation_fn")
        if "features_extractor_class" in policy_kwargs:
            features_extractor_class = policy_kwargs.pop("features_extractor_class")
        if "features_extractor_kwargs" in policy_kwargs:
            features_extractor_kwargs = policy_kwargs.pop("features_extractor_kwargs")
        if "dropout_p" in policy_kwargs:
            dropout_p = policy_kwargs.pop("dropout_p")
        
        # Set defaults only if not provided
        if net_arch is None:
            net_arch = [] if features_extractor_class == NatureCNN else [64, 64]
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        self.dropout_p = dropout_p
        self._net_arch = net_arch
        self._feat_cls = features_extractor_class
        self._feat_kwargs = features_extractor_kwargs
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            **policy_kwargs,
        )

    def make_q_net(self) -> nn.Module:
        return DropoutQNet(
            self.observation_space,
            self.action_space.n,
            self._feat_cls,
            self._feat_kwargs,
            self._net_arch,
            self.dropout_p,
        )

    @torch.no_grad()
    def mc_forward(self, obs: torch.Tensor, n: int = 20) -> torch.Tensor:
        self.q_net.set_training_mode(True)
        outs = [self.q_net(obs, training=True) for _ in range(n)]
        self.q_net.set_training_mode(False)
        return torch.stack(outs, 0)


# ───────────────────── Algorithm class ─────────────────────────── #

class MCDropoutDQN(DQN):
    policy_aliases: Dict[str, Type[MCDropoutQPolicy]] = {
        "MlpPolicy": MCDropoutQPolicy,
        "CnnPolicy": MCDropoutQPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[MCDropoutQPolicy]],
        env: Union[GymEnv, str],
        dropout_p: float = 0.2,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50_000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10_000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        **kwargs,
    ) -> None:
        if policy_kwargs is None:
            policy_kwargs = {}
        if isinstance(policy_kwargs, str):
            import ast
            policy_kwargs = ast.literal_eval(policy_kwargs)
        
        # Create a copy to avoid modifying the original
        policy_kwargs = policy_kwargs.copy()
        
        # Only update dropout_p if it's not already in policy_kwargs
        if "dropout_p" not in policy_kwargs:
            policy_kwargs.update({"dropout_p": dropout_p})

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            **kwargs,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self.q_net_target.train(False)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # --- current Q with dropout ---
            actions = replay_data.actions
            if actions.ndim == 1:
                actions = actions.unsqueeze(1)
            actions = actions.long()
            current_q_all = self.q_net(replay_data.observations, training=True)
            current_q = torch.gather(current_q_all, 1, actions).squeeze(1)

            # --- target Q with MC dropout ---
            with torch.no_grad():
                next_q = self.q_net_target(replay_data.next_observations)
                next_actions = next_q.argmax(dim=1, keepdim=True)
                target_samples = self.policy.mc_forward(replay_data.next_observations, n=20)
                target_means = target_samples.mean(dim=0)
                target_q = torch.gather(target_means, 1, next_actions).squeeze(1)
                target_q = replay_data.rewards.flatten() + (
                    1 - replay_data.dones.flatten()
                ) * self.gamma * target_q

            loss = nn.functional.smooth_l1_loss(current_q, target_q)

            self.policy.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            self._n_updates += 1
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

    @torch.no_grad()
    def predict_mc(self, obs: Any, n: int = 20) -> torch.Tensor:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self.policy.mc_forward(obs, n)


ALGOS: Dict[str, Any] = {"mcdropout_dqn": MCDropoutDQN}