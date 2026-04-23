from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bots.python.ppo.runtime.melee_env import ACTION_BRANCH_SIZES, OBSERVATION_DIM


@dataclass(slots=True)
class PPOBatchStats:
    policy_loss: float
    value_loss: float
    entropy: float


class MultiDiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBSERVATION_DIM, branch_sizes: tuple[int, ...] = ACTION_BRANCH_SIZES):
        super().__init__()
        self.branch_sizes = branch_sizes
        hidden = 256
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_heads = nn.ModuleList([nn.Linear(hidden, branch) for branch in branch_sizes])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        hidden = self.backbone(obs)
        logits = [head(hidden) for head in self.policy_heads]
        value = self.value_head(hidden).squeeze(-1)
        return logits, value


class PPOTrainer:
    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        branch_sizes: tuple[int, ...] = ACTION_BRANCH_SIZES,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        learning_rate: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.8,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.model = MultiDiscreteActorCritic(obs_dim=obs_dim, branch_sizes=branch_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.branch_sizes = branch_sizes

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> dict[str, np.ndarray | float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs_t)
        dists = [torch.distributions.Categorical(logits=branch_logits) for branch_logits in logits]
        actions = [dist.sample() for dist in dists]
        log_prob = sum(dist.log_prob(action) for dist, action in zip(dists, actions))
        entropy = sum(dist.entropy() for dist in dists)
        return {
            "action": np.array([action.item() for action in actions], dtype=np.int64),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(entropy.item()),
        }

    def update(
        self,
        rollout: dict[str, np.ndarray],
        epochs: int = 4,
        minibatch_size: int = 256,
    ) -> PPOBatchStats:
        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        for _ in range(epochs):
            indices = torch.randperm(obs.shape[0], device=self.device)
            for start in range(0, obs.shape[0], minibatch_size):
                batch_idx = indices[start : start + minibatch_size]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                logits, values = self.model(batch_obs)
                dists = [torch.distributions.Categorical(logits=branch_logits) for branch_logits in logits]
                log_probs = torch.stack(
                    [
                        dist.log_prob(batch_actions[:, branch_idx])
                        for branch_idx, dist in enumerate(dists)
                    ],
                    dim=0,
                ).sum(dim=0)
                entropy = torch.stack([dist.entropy() for dist in dists], dim=0).sum(dim=0).mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy.item())

        return PPOBatchStats(
            policy_loss=last_policy_loss,
            value_loss=last_value_loss,
            entropy=last_entropy,
        )

    @staticmethod
    def finish_rollout(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        bootstrap_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
            advantages[t] = last_advantage

        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)
